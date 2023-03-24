from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReactionMatrix:
    spec_r: np.ndarray
    spec_p: np.ndarray
    rate: np.ndarray


def process_reactions(reactions):
    spec_1st_r = []
    spec_1st_p = []
    rate_1st = []

    spec_2nd_r = []
    spec_2nd_p = []
    rate_2nd = []

    for reac, prod, rate in reactions:
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
            spec_1st_p.extend(prod)
            spec_1st_p.append(reac[0])
            spec_1st_r.extend(reac*(n_prod + 1))
            rate_1st.extend([rate]*(n_prod + 1))
            rate_1st[-1] = -rate_1st[-1]
        elif len(reac) == 2:
            spec_2nd_p.extend(prod)
            spec_2nd_p.extend(reac)
            spec_2nd_r.extend([reac]*(n_prod + 2))
            rate_2nd.extend([rate]*(n_prod + 2))
            rate_2nd[-1] = -rate_2nd[-1]
            rate_2nd[-2] = -rate_2nd[-2]
        else:
            raise ValueError
    
    rmat_1st = ReactionMatrix(
        np.asarray(spec_1st_r),
        np.asarray(spec_1st_p),
        np.asarray(rate_1st),
    )
    rmat_2nd = ReactionMatrix(
        np.asarray(spec_2nd_r).T,
        np.asarray(spec_2nd_p),
        np.asarray(rate_2nd),
    )
    return rmat_1st, rmat_2nd


def create_specie_table(rmat_1st, rmat_2nd):
    spec = np.concatenate((
        np.unique(rmat_1st.spec_r),
        np.unique(rmat_1st.spec_p),
        np.unique(rmat_2nd.spec_r[0]),
        np.unique(rmat_2nd.spec_r[1]),
        np.unique(rmat_2nd.spec_p),
    ))
    spec = np.unique(spec)
    table = pd.DataFrame(np.arange(len(spec)), index=spec, columns=['Index'])
    return table


def name_to_inds(spec_table, rmat_1st, rmat_2nd):
    inds_1st_r = spec_table.loc[rmat_1st.spec_r].values.ravel()
    inds_1st_p = spec_table.loc[rmat_1st.spec_p].values.ravel()
    rmat_1st = ReactionMatrix(inds_1st_r, inds_1st_p, rmat_1st.rate)

    inds_2nd_r0 = spec_table.loc[rmat_2nd.spec_r[0]].values.ravel()
    inds_2nd_r1 = spec_table.loc[rmat_2nd.spec_r[1]].values.ravel()
    inds_2nd_p = spec_table.loc[rmat_2nd.spec_p].values.ravel()
    rmat_2nd = ReactionMatrix((inds_2nd_r0, inds_2nd_r1), inds_2nd_p, rmat_2nd.rate)
    return rmat_1st, rmat_2nd
