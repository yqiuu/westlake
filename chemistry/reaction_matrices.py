from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class ReactionMatrix:
    order: int
    inds_id: np.ndarray
    inds_k: np.ndarray
    rate_sign: np.ndarray
    inds_r: np.ndarray
    inds_p: np.ndarray


def create_reaction_data(reactant_1, reactant_2, products, spec_table_base=None):
    rmat_1st, rmat_2nd = create_reaction_matrix(reactant_1, reactant_2, products)
    spec_table = create_specie_table(rmat_1st, rmat_2nd)
    if spec_table_base is not None:
        spec_table[spec_table_base.columns] = spec_table_base.loc[spec_table.index]
    rmat_1st, rmat_2nd = name_to_inds(spec_table, rmat_1st, rmat_2nd)
    return spec_table, rmat_1st, rmat_2nd


def create_reaction_matrix(reactant_1, reactant_2, products):
    inds_id_1st = []
    inds_k_1st = []
    rate_sign_1st = []
    inds_r_1st = []
    inds_p_1st = []

    inds_id_2nd = []
    inds_k_2nd = []
    rate_sign_2nd = []
    inds_r_2nd = []
    inds_p_2nd = []

    for idx, (reac_1, reac_2, prod) in enumerate(zip(reactant_1, reactant_2, products)):
        if reac_2 == "":
            reac = (reac_1,)
        else:
            reac = (reac_1, reac_2)

        if type(prod) is str:
            prod = prod.split(";")
        else:
            raise ValueError

        n_prod = len(prod)
        if len(reac) == 1:
            inds_id_1st.append(idx)
            inds_k_1st.extend([idx]*(n_prod + 1))
            rate_sign_1st.extend([1.]*(n_prod + 1))
            rate_sign_1st[-1] = -1.
            inds_p_1st.extend(prod)
            inds_p_1st.append(reac[0])
            inds_r_1st.extend(reac*(n_prod + 1))
        elif len(reac) == 2:
            inds_id_2nd.append(idx)
            inds_k_2nd.extend([idx]*(n_prod + 2))
            rate_sign_2nd.extend([1.]*(n_prod + 2))
            rate_sign_2nd[-1] = -1.
            rate_sign_2nd[-2] = -1.
            inds_p_2nd.extend(prod)
            inds_p_2nd.extend(reac)
            inds_r_2nd.extend([reac]*(n_prod + 2))
        else:
            raise ValueError

    rmat_1st = ReactionMatrix(
        order=1,
        inds_id=inds_id_1st,
        inds_k=inds_k_1st,
        rate_sign = rate_sign_1st,
        inds_r=inds_r_1st,
        inds_p=inds_p_1st,
    )
    rmat_2nd = ReactionMatrix(
        order=2,
        inds_id=inds_id_2nd,
        inds_k=inds_k_2nd,
        rate_sign=rate_sign_2nd,
        inds_r=inds_r_2nd,
        inds_p=inds_p_2nd,
    )
    return rmat_1st, rmat_2nd


def create_specie_table(rmat_1st, rmat_2nd):
    spec = np.concatenate((
        np.unique(rmat_1st.inds_r),
        np.unique(rmat_1st.inds_p),
        np.unique(rmat_2nd.inds_r),
        np.unique(rmat_2nd.inds_p),
    ))
    spec = np.unique(spec)
    table = pd.DataFrame(np.arange(len(spec)), index=spec, columns=['index'])
    table["index_grain"] = -1
    spec_grain = list(filter(lambda spec: spec.startswith("J") or spec.startswith("K"), spec))
    table.loc[spec_grain, "index_grain"] = np.arange(len(spec_grain))
    return table


def name_to_inds(spec_table, rmat_1st, rmat_2nd):
    inds_r_1st = spec_table.loc[rmat_1st.inds_r, "index"].values.ravel()
    inds_p_1st = spec_table.loc[rmat_1st.inds_p, "index"].values.ravel()
    rmat_1st = replace(rmat_1st, inds_r=inds_r_1st, inds_p=inds_p_1st)

    inds_r_2nd_ = np.asarray(rmat_2nd.inds_r)
    inds_r_2nd = np.zeros(inds_r_2nd_.shape, spec_table["index"].dtype)
    inds_r_2nd[:, 0] = spec_table.loc[inds_r_2nd_[:, 0], "index"].values.ravel()
    inds_r_2nd[:, 1] = spec_table.loc[inds_r_2nd_[:, 1], "index"].values.ravel()
    inds_p_2nd = spec_table.loc[rmat_2nd.inds_p, "index"].values.ravel()
    rmat_2nd = replace(rmat_2nd, inds_r=inds_r_2nd, inds_p=inds_p_2nd)
    return rmat_1st, rmat_2nd