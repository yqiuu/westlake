from dataclasses import dataclass, replace

import numpy as np


@dataclass
class ReactionMat:
    """A set of parameters that define the kinetic equation.

    This is used as a dataclass for `ReactionMatrix`.

    Args:
        order (int): Number of reactants.
        n_spec (int): Number of species including both first and second order
            reactions.
        inds_id (array): Index in the reaction dataframe.
        inds_k (array): Index in the reaction dataframe for computing the rates.
        rate_sign (array): Sign of the rates. This aligns with ``inds_k``.
        inds_r (array): Index in the specie table for gathering the reactants.
            For second order reactions, each element is a pair of indices.
        inds_p (array): Index in the specie table for gathering the products.
    """
    order: int
    n_spec: int
    inds_id_uni: np.ndarray
    inds_id: np.ndarray
    inds_k: np.ndarray
    rate_sign: np.ndarray
    inds_r: np.ndarray
    inds_p: np.ndarray

    def extract(self, cond, use_id_uni):
        if use_id_uni:
            inds_id_uni = self.inds_id_uni[cond]
            tmp = set(inds_id_uni)
            cond = [idx in tmp for idx in self.inds_id]
        else:
            inds_id_uni = self.inds_id_uni
        inds_id = self.inds_id[cond]
        inds_k = self.inds_k[cond]
        if self.rate_sign is None:
            rate_sign = None
        else:
            rate_sign = self.rate_sign[cond]
        inds_r = self.inds_r[cond]
        inds_p = self.inds_p[cond]
        return ReactionMat(
            self.order, self.n_spec, inds_id_uni,
            inds_id, inds_k, rate_sign, inds_r, inds_p
        )

    def split(self, cond, use_id_uni):
        return self.extract(cond, use_id_uni), self.extract(~cond, use_id_uni)


class ReactionMatrix:
    def __init__(self, df_reac, df_spec):
        _, self._rmat_1st, self._rmat_2nd = self._derive_indices(
            df_reac.index.values,
            df_reac["reactant_1"].values,
            df_reac["reactant_2"].values,
            df_reac["products"].values,
        )
        self._df_spec = df_spec

    def create_index_matrices(self):
        """Convert specie names to indices for both reaction matrices."""
        df_spec = self._df_spec

        rmat_1st = self._rmat_1st
        if len(rmat_1st.inds_id) > 0:
            inds_r_1st = df_spec.index.get_indexer(rmat_1st.inds_r)
            inds_p_1st = df_spec.index.get_indexer(rmat_1st.inds_p)
            rmat_1st = replace(rmat_1st, inds_r=inds_r_1st, inds_p=inds_p_1st)
        else:
            rmat_1st = None

        rmat_2nd = self._rmat_2nd
        if len(rmat_2nd.inds_id) > 0:
            inds_r_2nd_ = np.asarray(rmat_2nd.inds_r)
            inds_r_2nd = np.zeros(inds_r_2nd_.shape, dtype="i8")
            inds_r_2nd[:, 0] = df_spec.index.get_indexer(inds_r_2nd_[:, 0])
            inds_r_2nd[:, 1] = df_spec.index.get_indexer(inds_r_2nd_[:, 1])
            inds_p_2nd = df_spec.index.get_indexer(rmat_2nd.inds_p)
            rmat_2nd = replace(rmat_2nd, inds_r=inds_r_2nd, inds_p=inds_p_2nd)
        else:
            rmat_2nd = None

        return rmat_1st, rmat_2nd

    def _derive_indices(self, indices, reactant_1, reactant_2, products):
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

        for idx, reac_1, reac_2, prod in zip(indices, reactant_1, reactant_2, products):
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
            inds_id_uni=np.array(inds_id_1st),
            inds_id=np.array(inds_k_1st),
            inds_k=np.array(inds_k_1st),
            rate_sign=np.array(rate_sign_1st),
            inds_r=np.array(inds_r_1st),
            inds_p=np.array(inds_p_1st),
        )
        rmat_2nd = ReactionMat(
            order=2,
            n_spec=n_spec,
            inds_id_uni=np.array(inds_id_2nd),
            inds_id=np.array(inds_k_2nd),
            inds_k=np.array(inds_k_2nd),
            rate_sign=np.array(rate_sign_2nd),
            inds_r=np.array(inds_r_2nd),
            inds_p=np.array(inds_p_2nd),
        )
        return species, rmat_1st, rmat_2nd