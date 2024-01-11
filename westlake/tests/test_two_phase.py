import pickle
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import torch
import westlake

from utils import get_abs_fname

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
FILE_NAME = "two_phase.pickle"
RTOL = 1e-5
ATOL = 1e-20


def test_two_phase():
    ab_dict_new = solve_two_phase()
    ab_dict_fid = pickle.load(open(get_abs_fname(FILE_NAME), "rb"))
    for ab_new, ab_fid in zip(ab_dict_new.values(), ab_dict_fid.values()):
        assert_allclose(ab_new, ab_fid, rtol=RTOL, atol=ATOL)


def solve_two_phase():
    dirname = get_dirname()
    df_reac = pd.read_csv(dirname/Path("reactions.csv"), na_filter=False)
    df_spec = pd.read_csv(dirname/Path("species.csv"), index_col="specie")
    df_surf = pd.read_csv(dirname/Path("surface_parameters.csv"), index_col="specie")

    # Initial condition
    ab_0_dict = {
        'H': 0.0,
        'H2': 0.5,
        'He': 0.09,
        'C+': 1.7e-04,
        'S+': 8e-08,
    }
    # Create config
    config = westlake.Config(
        model="two phase",
        dtg_mass_ratio=westlake.fixed_dtg_mass_ratio(ab_0_dict['He']),
        H2_shielding="Lee+1996",
        CO_shielding="Lee+1996",
        Av=2.7,
        den_gas=1.3e4,
        T_gas=11.2,
        T_dust=15.3,
        uv_flux=1.9,
        t_start=1e-10,
        t_end=1e4,
        use_scipy_solver=True,
        method="LSODA",
        atol=ATOL,
    )

    n_eval = 100
    t_eval = np.logspace(np.log10(config.t_start), np.log10(config.t_end), n_eval)
    reac_term = westlake.create_astrochem_model(df_reac, df_spec, df_surf, config)
    res = westlake.solve_rate_equation_astrochem(
        reac_term, ab_0_dict, df_spec, config, t_eval=t_eval
    )

    ab_dict = {"time": res["time"]}
    for spec, ab in zip(df_spec.index, res.ab):
        ab_dict[spec] = ab
    return ab_dict


def get_dirname():
    return Path(__file__).parent/Path("inputs")


if __name__ == "__main__":
    # Save test data
    ab_dict = solve_two_phase()
    pickle.dump(ab_dict, open(FILE_NAME, "wb"))
