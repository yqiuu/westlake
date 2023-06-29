from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class ReactionMat:
    """A set of parameters that define the kinetic equation.

    This is used as a dataclass for `ReactionMatrix`.

    Args:
        order (int): Number of reactants.
        n_spec (int): Number of species including both first and second order
            reactions.
        inds_id (list): Index in the reaction dataframe.
        inds_k (list): Index in the reaction dataframe for computing the rates.
        rate_sign (list): Sign of the rates. This aligns with ``inds_k``.
        inds_r (list): Index in the specie table for gathering the reactants.
            For second order reactions, each element is a pair of indices.
        inds_p (list): Index in the specie table for gathering the products.
    """
    order: int
    n_spec: int
    inds_id: list
    inds_k: list
    rate_sign: list
    inds_r: list
    inds_p: list


class ReactionMatrix:
    def __init__(self, reactant_1, reactant_2, products, df_spec=None):
        species, self._rmat_1st, self._rmat_2nd \
            = self._derive_indices(reactant_1, reactant_2, products)
        df_spec_new = self._create_specie_table(species)
        if df_spec is not None:
            df_spec_new[df_spec.columns] = df_spec.loc[df_spec_new.index]
        self._df_spec = df_spec_new

    @property
    def df_spec(self):
        return self._df_spec

    def create_index_matrices(self):
        """Convert specie names to indices for both reaction matrices."""
        df_spec = self._df_spec
        rmat_1st = self._rmat_1st
        inds_r_1st = df_spec.loc[rmat_1st.inds_r, "index"].values.ravel()
        inds_p_1st = df_spec.loc[rmat_1st.inds_p, "index"].values.ravel()
        rmat_1st = replace(rmat_1st, inds_r=inds_r_1st, inds_p=inds_p_1st)

        rmat_2nd = self._rmat_2nd
        inds_r_2nd_ = np.asarray(rmat_2nd.inds_r)
        inds_r_2nd = np.zeros(inds_r_2nd_.shape, df_spec["index"].dtype)
        inds_r_2nd[:, 0] = df_spec.loc[inds_r_2nd_[:, 0], "index"].values.ravel()
        inds_r_2nd[:, 1] = df_spec.loc[inds_r_2nd_[:, 1], "index"].values.ravel()
        inds_p_2nd = df_spec.loc[rmat_2nd.inds_p, "index"].values.ravel()
        rmat_2nd = replace(rmat_2nd, inds_r=inds_r_2nd, inds_p=inds_p_2nd)
        return rmat_1st, rmat_2nd

    def _derive_indices(self, reactant_1, reactant_2, products):
        species = set()

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
            if type(prod) is str:
                prod = prod.split(";")
            else:
                raise ValueError("Unknown products.")

            species.add(reac_1)
            species.update(set(prod))

            n_prod = len(prod)
            if reac_2 == "":
                inds_id_1st.append(idx)
                inds_k_1st.extend([idx]*(n_prod + 1))
                rate_sign_1st.extend([1.]*(n_prod + 1))
                rate_sign_1st[-1] = -1.
                inds_p_1st.extend(prod)
                inds_p_1st.append(reac_1)
                inds_r_1st.extend([reac_1]*(n_prod + 1))
            else:
                species.add(reac_2)

                inds_id_2nd.append(idx)
                inds_k_2nd.extend([idx]*(n_prod + 2))
                rate_sign_2nd.extend([1.]*(n_prod + 2))
                rate_sign_2nd[-1] = -1.
                rate_sign_2nd[-2] = -1.
                inds_p_2nd.extend(prod)
                reac = [reac_1, reac_2]
                inds_p_2nd.extend(reac)
                inds_r_2nd.extend([reac]*(n_prod + 2))

        n_spec = len(species)
        rmat_1st = ReactionMat(
            order=1,
            n_spec=n_spec,
            inds_id=inds_id_1st,
            inds_k=inds_k_1st,
            rate_sign = rate_sign_1st,
            inds_r=inds_r_1st,
            inds_p=inds_p_1st,
        )
        rmat_2nd = ReactionMat(
            order=2,
            n_spec=n_spec,
            inds_id=inds_id_2nd,
            inds_k=inds_k_2nd,
            rate_sign=rate_sign_2nd,
            inds_r=inds_r_2nd,
            inds_p=inds_p_2nd,
        )
        return species, rmat_1st, rmat_2nd

    def _create_specie_table(self, species):
        species = list(species)
        species.sort()
        return pd.DataFrame(np.arange(len(species)), index=species, columns=['index'])