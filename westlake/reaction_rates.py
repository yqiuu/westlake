from collections import defaultdict
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from torch import nn

from .utils import data_frame_to_tensor_dict
from .equation import ReactionTerm


class ConstantReactionRate(nn.Module):
    def __init__(self, rmat, rate):
        super(ConstantReactionRate, self).__init__()
        rate = torch.tensor(rmat.rate_sign*rate[rmat.inds_k], dtype=torch.get_default_dtype())
        self.register_buffer("rate", rate)

    def forward(self, t_in, params_med=None):
        return self.rate


def create_constant_rate_model(reaction_matrix, df_reac):
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()
    rate_1st = ConstantReactionRate(rmat_1st, df_reac["rate"].values)
    rate_2nd = ConstantReactionRate(rmat_2nd, df_reac["rate"].values)
    # The rate signs are included in the rates, and therefore, they are
    # unnecessary.
    rmat_1st.rate_sign = None
    rmat_2nd.rate_sign = None
    return ReactionTerm(rmat_1st, rate_1st, rmat_2nd, rate_2nd)


class FormulaDictReactionModule(nn.Module):
    def __init__(self, rmat, formula_dict, inds_fm_dict, inds_reac, params_reac):
        super(FormulaDictReactionModule, self).__init__()
        self.order = rmat.order
        for i_fm, inds in enumerate(inds_fm_dict.values()):
            setattr(self, f"_params_reac_{i_fm}", params_reac.indexing(inds))
        self.formula_list = nn.ModuleList([formula_dict[key] for key in inds_fm_dict])
        self.register_buffer(
            "rate_sign", torch.tensor(rmat.rate_sign, dtype=torch.get_default_dtype()))
        self.register_buffer("inds_reac", torch.tensor(inds_reac))

    def forward(self, t_in, params_med):
        batch_size = next(iter(params_med.values())).shape[0]
        def compute_rates_sub(i_fm):
            params_reac_sub = getattr(self, f"_params_reac_{i_fm}")()
            rates = self.formula_list[i_fm](params_med, params_reac_sub)
            if rates.dim() == 1:
                rates = rates.repeat(batch_size, 1)
            return rates

        return torch.concat(
            [compute_rates_sub(i_fm) for i_fm in range(len(self.formula_list))], dim=-1)

    def compute_rates_reac(self, t_in, params_med):
        # out: (B, R)
        return self.forward(t_in, params_med)[:, self.inds_reac]


class FormulaDictReactionFactory:
    def __init__(self, df_reac, rmat):
        # The code below construct the following variables.
        #   1. inds_reac, index in the reaction dataframe for the outputs of the reaction module.
        #   2. inds_k, index of the rates in the equation for the outputs of the reaction module.
        #   This should align with rmat.rate_sign

        df_sub = df_reac.iloc[rmat.inds_id]
        # Link the indices in the df_sub to those in the reaction dataframe.
        lookup_sub = pd.DataFrame(np.arange(len(df_sub)), index=df_sub.index, columns=["index_sub"])
        # Link the indices in the outputs of the reaction module to those in the df_sub.
        inds_fm_dict = defaultdict(list)
        for i_fm, fm in enumerate(df_sub["formula"]):
            inds_fm_dict[fm].append(i_fm)
        inds_fm = np.asarray(sum(inds_fm_dict.values(), start=[]))
        # Link the indices in the outputs of the reaction module to those in the reaction dataframe.
        lookup_fm = pd.DataFrame(np.arange(len(inds_fm)), index=inds_fm, columns=["index_fm"])
        lookup_sub["index_fm"] = lookup_fm.loc[lookup_sub["index_sub"], "index_fm"].values\

        inds_reac = lookup_sub.loc[rmat.inds_id, "index_fm"].values
        inds_k = lookup_sub.loc[rmat.inds_k, "index_fm"].values

        # Construct indices for each formula
        inds_fm_sub = []
        for inds in inds_fm_dict.values():
            if len(inds) != 0:
                inds_fm_sub.extend(list(range(len(inds))))
        lookup_sub["index_fm_sub"] = inds_fm_sub

        # Set attributes
        self._df_sub = df_sub
        self._rmat = rmat
        self._lookup = lookup_sub
        self._inds_fm_dict = inds_fm_dict
        self._inds_reac = inds_reac
        self._inds_k = inds_k

    def create(self, formula_dict, param_names):
        """Create a FormulaDictReactionModule.

        Args:
            formula_dict (dict): A dictionary of formulae to compute reaction
                rates.
            param_names (list): Parameter names in the reaction dataframe that
                are required when computing the reaction rates.

        Returns:
            FormulaDictReactionModule: reaction module.
        """
        return FormulaDictReactionModule(
            self._rmat, formula_dict, self._inds_fm_dict, self._inds_reac, self._inds_k,
            data_frame_to_tensor_dict(self._df_sub[param_names])
        )

    def find_indices_in_formula(self, formula, condition):
        """Find indices of some specific reactions in the a reaction rate
        module.

        This is used to deal with special cases in reactions.

        Args:
            formula (str): Formula name in the reaction dataframe.
            condition (callable): A callable that takes the reaction dataframe
                and returns a bool array to to select reactions.

        Raises:
            KeyError: If no reactions are found.

        Returns:
            int | list: A list of indices or an integer if there is only one
                matched reaction.
        """
        df_sub = self._df_sub
        cond = (df_sub["formula"] == formula) & condition(df_sub)
        indices = self._lookup.loc[df_sub.index[cond], "index_fm_sub"].values
        n_ind = len(indices)
        if n_ind == 0:
            raise KeyError("Cannot find the indices that satisfy the condition.")
        elif n_ind == 1:
            return indices[0]
        return list(indices)


def create_formula_dict_reaction_module(df_reac, rmat, formula_dict, param_names):
    df_sub = df_reac.iloc[rmat.inds_id]

    # The code below construct the following variables.
    #   1. inds_reac, index in the reaction dataframe for the outputs of the reaction module.
    #   2. inds_k, index of the rates in the equation for the outputs of the reaction module. This
    #   should align with rmat.rate_sign

    # Link the indices in the df_sub to those in the reaction dataframe.
    lookup_sub = pd.DataFrame(np.arange(len(df_sub)), index=df_sub.index, columns=["index_sub"])
    # Link the indices in the outputs of the reaction module to those in the df_sub.
    inds_fm_dict = defaultdict(list)
    for i_fm, fm in enumerate(df_sub["formula"]):
        inds_fm_dict[fm].append(i_fm)
    inds_fm = np.asarray(sum(inds_fm_dict.values(), start=[]))
    lookup_fm = pd.DataFrame(np.arange(len(inds_fm)), index=inds_fm, columns=["index_fm"])
    # Link the indices in the outputs of the reaction module to those in the reaction dataframe.
    lookup_sub["index_fm"] = lookup_fm.loc[lookup_sub["index_sub"], "index_fm"].values
    #
    inds_reac = lookup_sub.loc[rmat.inds_id, "index_fm"].values
    inds_k = lookup_sub.loc[rmat.inds_k, "index_fm"].values

    rmod = FormulaDictReactionModule(
        rmat, formula_dict, inds_fm_dict, inds_reac,
        data_frame_to_tensor_dict(df_sub[param_names])
    )
    rmat_new = replace(rmat, inds_k=inds_k)
    return rmod, rmat_new