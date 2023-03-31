from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReactionMatrix:
    inds: object
    rate_sign: object
    spec_r: object
    spec_p: object


def create_reaction_data(reactants, products):
    rmat_1st, rmat_2nd = create_reaction_matrix(reactants, products)
    spec_table = create_specie_table(rmat_1st, rmat_2nd)
    rmat_1st, rmat_2nd = name_to_inds(spec_table, rmat_1st, rmat_2nd)
    return spec_table, rmat_1st, rmat_2nd


def create_reaction_matrix(reactants, products):
    inds_1st = []
    rate_1st = []
    spec_1st_r = []
    spec_1st_p = []

    inds_2nd = []
    rate_2nd = []
    spec_2nd_r = []
    spec_2nd_p = []

    for idx, (reac, prod) in enumerate(zip(reactants, products)):
        if type(reac) is str:
            reac = reac.split(";")
        else:
            raise ValueError

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
        np.asarray(inds_1st),
        np.asarray(rate_1st),
        np.asarray(spec_1st_r),
        np.asarray(spec_1st_p),
    )
    rmat_2nd = ReactionMatrix(
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
    table = pd.DataFrame(np.arange(len(spec)), index=spec, columns=['Index'])
    table["IndexGrain"] = -1
    spec_grain = list(filter(lambda spec: spec.startswith("J") or spec.startswith("K"), spec))
    table.loc[spec_grain, "IndexGrain"] = np.arange(len(spec_grain))
    return table


def name_to_inds(spec_table, rmat_1st, rmat_2nd):
    inds_1st_r = spec_table.loc[rmat_1st.spec_r, "Index"].values.ravel()
    inds_1st_p = spec_table.loc[rmat_1st.spec_p, "Index"].values.ravel()
    rmat_1st = ReactionMatrix(rmat_1st.inds, rmat_1st.rate_sign, inds_1st_r, inds_1st_p)

    inds_2nd_r = np.zeros(rmat_2nd.spec_r.shape, spec_table["Index"].dtype)
    inds_2nd_r[:, 0] = spec_table.loc[rmat_2nd.spec_r[:, 0], "Index"].values.ravel()
    inds_2nd_r[:, 1] = spec_table.loc[rmat_2nd.spec_r[:, 1], "Index"].values.ravel()
    inds_2nd_p = spec_table.loc[rmat_2nd.spec_p, "Index"].values.ravel()
    rmat_2nd = ReactionMatrix(rmat_2nd.inds, rmat_2nd.rate_sign, inds_2nd_r, inds_2nd_p)
    return rmat_1st, rmat_2nd
