from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class ReactionMatrix:
    order: int
    inds: object
    rate_sign: object
    spec_r: object
    spec_p: object


def create_reaction_data(reactant_1, reactant_2, products, spec_table_base=None):
    rmat_1st, rmat_2nd = create_reaction_matrix(reactant_1, reactant_2, products)
    spec_table = create_specie_table(rmat_1st, rmat_2nd)
    if spec_table_base is not None:
        spec_table[spec_table_base.columns] = spec_table_base.loc[spec_table.index]
    rmat_1st, rmat_2nd = name_to_inds(spec_table, rmat_1st, rmat_2nd)
    return spec_table, rmat_1st, rmat_2nd


def create_reaction_matrix(reactant_1, reactant_2, products):
    inds_1st = []
    rate_1st = []
    spec_1st_r = []
    spec_1st_p = []

    inds_2nd = []
    rate_2nd = []
    spec_2nd_r = []
    spec_2nd_p = []

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
            inds_1st.extend([idx]*(n_prod + 1))
            rate_1st.extend([1.]*(n_prod + 1))
            rate_1st[-1] = -1.
            spec_1st_p.extend(prod)
            spec_1st_p.append(reac[0])
            spec_1st_r.extend(reac*(n_prod + 1))
        elif len(reac) == 2:
            inds_2nd.extend([idx]*(n_prod + 2))
            rate_2nd.extend([1.]*(n_prod + 2))
            rate_2nd[-1] = -1.
            rate_2nd[-2] = -1.
            spec_2nd_p.extend(prod)
            spec_2nd_p.extend(reac)
            spec_2nd_r.extend([reac]*(n_prod + 2))
        else:
            raise ValueError

    rmat_1st = ReactionMatrix(
        1, # reaction order
        np.asarray(inds_1st),
        np.asarray(rate_1st),
        np.asarray(spec_1st_r),
        np.asarray(spec_1st_p),
    )
    rmat_2nd = ReactionMatrix(
        2, # reaction order
        np.asarray(inds_2nd),
        np.asarray(rate_2nd),
        np.asarray(spec_2nd_r),
        np.asarray(spec_2nd_p),
    )
    return rmat_1st, rmat_2nd


def create_specie_table(rmat_1st, rmat_2nd):
    spec = np.concatenate((
        np.unique(rmat_1st.spec_r),
        np.unique(rmat_1st.spec_p),
        np.unique(rmat_2nd.spec_r),
        np.unique(rmat_2nd.spec_p),
    ))
    spec = np.unique(spec)
    table = pd.DataFrame(np.arange(len(spec)), index=spec, columns=['index'])
    table["index_grain"] = -1
    spec_grain = list(filter(lambda spec: spec.startswith("J") or spec.startswith("K"), spec))
    table.loc[spec_grain, "index_grain"] = np.arange(len(spec_grain))
    return table


def name_to_inds(spec_table, rmat_1st, rmat_2nd):
    inds_1st_r = spec_table.loc[rmat_1st.spec_r, "index"].values.ravel()
    inds_1st_p = spec_table.loc[rmat_1st.spec_p, "index"].values.ravel()
    rmat_1st = replace(rmat_1st, spec_r=inds_1st_r, spec_p=inds_1st_p)

    inds_2nd_r = np.zeros(rmat_2nd.spec_r.shape, spec_table["index"].dtype)
    inds_2nd_r[:, 0] = spec_table.loc[rmat_2nd.spec_r[:, 0], "index"].values.ravel()
    inds_2nd_r[:, 1] = spec_table.loc[rmat_2nd.spec_r[:, 1], "index"].values.ravel()
    inds_2nd_p = spec_table.loc[rmat_2nd.spec_p, "index"].values.ravel()
    rmat_2nd = replace(rmat_2nd, spec_r=inds_2nd_r, spec_p=inds_2nd_p)
    return rmat_1st, rmat_2nd