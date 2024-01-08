import yaml
from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import westlake


def main(dirname, use_evolution):
    df_reac, df_spec, df_surf, df_barr, df_ma, df_act, ab_0_dict = \
        load_inputs_nautilus(dirname)
    config_dict = yaml.safe_load(open(Path(dirname)/Path("config.yml")))

    if use_evolution:
        config_dict.update(atol=1e-30)
    else:
        config_dict.update(atol=1e-23)
    dtg_mass_ratio = westlake.fixed_dtg_mass_ratio(
        ab_0_dict.get("He", 0.), config_dict.get("dtg_mass_ratio", None))
    config_dict.update(dtg_mass_ratio=dtg_mass_ratio)
    config = westlake.Config(**config_dict)

    if use_evolution:
        fname = Path(dirname)/Path("structure_evolution.dat")
        data_med = np.loadtxt(fname, comments="!")
        t_end = data_med[-1, 0] # yr
        data_med[:, 0] = data_med[:, 0]*config.to_second # sec
        data_med[:, 1:] = 10**data_med[:, 1:]
        data_med = torch.as_tensor(data_med, dtype=torch.get_default_dtype())
        x_med = data_med[:, 0].contiguous()
        medium = westlake.Medium(
            config,
            Av=westlake.LinearInterpolation(x_med, data_med[:, 1:2]),
            den_gas=westlake.LinearInterpolation(x_med, data_med[:, 2:3]),
            T_gas=westlake.LinearInterpolation(x_med, data_med[:, 3:4]),
            T_dust=westlake.LinearInterpolation(x_med, data_med[:, 3:4]),
        )
    else:
        # Use Av, den_gas, T_gas, T_dust defined in confg
        medium = None
        t_end = config.t_end

    reaction_term = westlake.create_astrochem_model(
        df_reac, df_spec, df_surf, config,
        medium=medium, df_act=df_act, df_barr=df_barr, df_ma=df_ma,
    )
    res = westlake.solve_rate_equation_astrochem(
        reaction_term, ab_0_dict, df_spec, config,
        t_span=(config.t_start, t_end)
    )
    save_name = Path(dirname)/Path("res.pickle")
    westlake.save_result(res, save_name)


def load_inputs_nautilus(dirname):
    """Load inputs that adopt the Nautilus format.

    Args:
        dirname (str): Directory name.

    Returns:
        df_reac (pd.DataFrame): Reaction network.
        df_spec (pd.DataFrame): Specie table.
        df_surf (pd.DataFrame): Surface parameters.
        df_barr (pd.DataFrame): Extra diffusion energy.
        df_ma (pd.DataFrame): Extra molecular weight.
        df_act (pd.DataFrame): Acitviation energy.
        ab_0_dict (dict): Initial abundances
    """
    dirname = Path(dirname)
    df_reac = load_reaction_network(
        dirname/Path("gas_reactions.in"),
        dirname/Path("grain_reactions.in")
    )
    df_spec = load_species(
        dirname/Path("gas_species.in"),
        dirname/Path("grain_species.in"),
        dirname/Path("element.in")
    )
    df_surf, df_barr, df_ma = load_surface_params(dirname/Path("surface_parameters.in"))
    df_act = load_activation_energies(dirname/Path("activation_energies.in"))
    ab_0_dict = load_initial_abundances(dirname/Path("abundances.in"))
    return df_reac, df_spec, df_surf, df_barr, df_ma, df_act, ab_0_dict


def load_reaction_network(file_gas, file_grain):
    lines = open(file_gas).readlines()
    lines.extend(open(file_grain).readlines())

    n_col_reac = 34
    n_col_prod = 91
    data = preprocess_lines(lines[1:], n_col_reac, n_col_prod)
    columns = [
        "key", "reactant_1", "reactant_2", "products", "alpha", "beta", "gamma",
        "X0", "X1", "logn", "ITYPE", "T_min", "T_max", "formula",
        "ID", "X2", "X3"
    ]
    df = pd.DataFrame(data, columns=columns)
    df.dropna(inplace=True)
    df = df[df["reactant_1"].map(lambda name: not name.startswith("!"))]

    df[["ID", "ITYPE", "formula"]] = df[["ID", "ITYPE", "formula"]].astype("i4")
    df[["T_min", "T_max", "alpha", "beta", "gamma"]] \
        = df[["T_min", "T_max", "alpha", "beta", "gamma"]].astype("f8")

    cond = (df["ITYPE"] >= 4) & (df["ITYPE"] <= 8)
    replace_dict = {
        0: "gas grain",
        1: "CR dissociation", 2: "CRP dissociation", 3: "photodissociation",
        10: "surface H2 formation", 11: "surface H accretion",
        14: "surface reaction",
        15: "thermal evaporation", 16: "CR evaporation",
        17: "CR dissociation", 18: "CR dissociation",
        19: "photodissociation", 20: "photodissociation",
        30: "Eley-Rideal",
        31: "complexes reaction",
        40: "surface to mantle", 41: "mantle to surface", 66: "UV photodesorption",
        67: "CR photodesorption", 99: "surface accretion",
    }
    df.loc[~cond, "formula"] = df.loc[~cond, "ITYPE"].replace(replace_dict)
    replace_dict = {
        3: "modified Arrhenius", 4: "ionpol1", 5: "ionpol2",
    }
    df.loc[cond, "formula"] = df.loc[cond, "formula"].replace(replace_dict)

    df.reset_index(drop=True, inplace=True)
    df.drop(columns=["ITYPE", "logn", "X0", "X1", "X2", "X3"], inplace=True)
    return df


def preprocess_lines(lines, n_col_reac, n_col_prod, remove_list=None):
    remove_list = [] if remove_list is None else remove_list
    for i_l in range(len(lines)):
        line = lines[i_l]
        for str_remove in ["XXX"] + remove_list:
            line = line.replace(str_remove, " "*len(str_remove))
        str_prod = ";".join(line[n_col_reac:n_col_prod].split())
        reacs = line[:n_col_reac].split()
        if len(reacs) == 1:
            str_reac_1 = reacs[0]
            str_reac_2 = ""
            key = f"{str_reac_1}>{str_prod}"
        elif len(reacs) == 2:
            str_reac_1, str_reac_2 = reacs
            key = f"{str_reac_1};{str_reac_2}>{str_prod}"
        params = line[n_col_prod:].split()
        lines[i_l] = [key, str_reac_1, str_reac_2, str_prod] + params
    return lines


def load_species(file_gas, file_grain, file_element):
    data = []
    for ln in open(file_gas).readlines():
        if ln.startswith("!"):
            continue
        data.append(ln.split())
    if file_grain is not None:
        for ln in open(file_grain).readlines():
            if ln.startswith("!"):
                continue
            data.append(ln.split())
    columns = [
        "specie", "charge",
        "H", "He", "C", "N", "O", "Si", "S", "Fe", "Na", "Mg", "Cl", "P", "F"
    ]
    df = pd.DataFrame(data, columns=columns)
    df.set_index("specie", inplace=True)
    df = df.astype("i4")

    data = []
    for ln in open(file_element).readlines():
        if ln.startswith("!"):
            continue
        data.append(ln.split())
    names, mas = list(zip(*data))
    df["num_atoms"] = np.sum(df[list(names)].values, axis=-1)
    df["ma"] = np.sum(df[list(names)].values*np.asarray(mas, dtype="f8"), axis=-1)
    return df


def load_surface_params(fname):
    index = []
    data = []
    for ln in open(fname).readlines():
        if ln.startswith("!"):
            continue
        ln_1 = ln.split()
        index.append(ln_1[0])
        data.append(ln_1[1:5] + ln[63:].split()[:1])
    columns = ["ma", "E_deso", "E_barr", "dE_band", "dHf"]
    df_surf = pd.DataFrame(data, index=index, columns=columns, dtype="f8")
    df_surf.loc[df_surf["dHf"] == -999.99, "dHf"] = np.nan

    df_barr = df_surf.loc[["JH"], ["E_barr"]].copy()
    df_ma = df_surf[["ma"]].copy()

    df_surf.drop(columns=["ma", "E_barr", "dE_band"], inplace=True)
    return df_surf, df_barr, df_ma


def load_activation_energies(fname):
    n_col_reac = 34
    n_col_prod_b = 37
    n_col_prod_e = 93
    data = []
    for ln in open(fname).readlines():
        if ln.startswith("!") or ln == '\n':
            continue
        str_prod = ";".join(ln[n_col_prod_b:n_col_prod_e].split())
        reacs = ln[:n_col_reac].split()
        if len(reacs) == 1:
            str_reac_1 = reacs[0]
            str_reac_2 = ""
            key = f"{str_reac_1}>{str_prod}"
        elif len(reacs) == 2:
            str_reac_1, str_reac_2 = reacs
            key = f"{str_reac_1};{str_reac_2}>{str_prod}"
        params = ln[n_col_prod_e:].split()[0]
        data.append([key, str_reac_1, str_reac_2, str_prod, params])
    columns = ["key", "reactant_1", "reactant_2", "products", "E_act"]
    df = pd.DataFrame(data, columns=columns)
    df["E_act"] = df["E_act"].astype("f8")
    df.set_index("key", inplace=True)
    return df


def load_initial_abundances(fname):
    ab_0_dict = {}
    for line in open(fname).readlines()[1:]:
        line = line.split('=')
        name = line[0].strip()
        val = line[1].split('!')[0].strip().replace("D", "e")
        val = float(val)
        ab_0_dict[name] = val
    return ab_0_dict


if __name__ == "__main__":
    parser = ArgumentParser(description='Run nautilus network.')
    parser.add_argument(
        '--dirname', type=str, default='./', help='Directory of the reaction network')
    parser.add_argument(
        '--use_evolution', action='store_true', help="Use structure_evolution.dat")
    args = parser.parse_args()
    main(args.dirname, args.use_evolution)