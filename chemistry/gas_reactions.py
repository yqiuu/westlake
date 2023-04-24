import torch
from torch import nn
from torch.nn import functional as F

from .utils import data_frame_to_tensor_dict
from .reaction_rates import FormulaDictReactionRate


class CosmicRayIonization(nn.Module):
    """Cosmic-ray ionization.

    Args:
        rate_cr_ion (float): H2 cosmic-ray ionization rate [s^-1].
    """
    def __init__(self, rate_cr_ion):
        super(CosmicRayIonization, self).__init__()
        self.register_buffer("rate_cr_ion", torch.tensor(rate_cr_ion, dtype=torch.float32))

    def forward(self, params_env, params_reac):
        """
        Args:
            params_reac (KeyTensor): (R, 5). Reaction parameters.
            params_env (KeyTensor): (3,). Environment parameters.

        Returns:
            tensor: (R,). Reaction rate.
        """
        rate = params_reac["alpha"]*self.rate_cr_ion
        return rate


class Photodissociation(nn.Module):
    def forward(self, params_env, params_reac):
        """
        Args:
            params_reac (KeyTensor): (R, 5). Reaction parameters.
            params_env (KeyTensor): (3,). Environment parameters.

        Returns:
            tensor: (R,). Reaction rate.
        """
        Av = params_env["Av"]
        if len(Av) != 1:
            Av = Av[:, None]
        rate = params_reac["alpha"]*torch.exp(-params_reac["gamma"]*Av)
        return rate


class ModifiedArrhenius(nn.Module):
    def forward(self, params_env, params_reac):
        """
        Args:
            params_reac (KeyTensor): (R, 5). Reaction parameters.
            params_env (KeyTensor): (3,). Environment parameters.

        Returns:
            tensor: (B, R). Reaction rate.
        """
        T_min, T_max, alpha, beta, gamma = params_reac.values()
        T_gas = params_env["T_gas"]
        if len(T_gas) != 1:
            T_gas = T_gas[:, None]
        # TODO: Check how to compute the rate if the temperature is beyond the
        # range.
        cond = (T_gas >= T_min) & (T_gas < T_max)
        cond = cond.type(T_gas.dtype)
        rate = alpha*(T_gas/300.)**beta*torch.exp(-gamma/T_gas)*cond
        return rate
    

class Ionpol1(nn.Module):
    def forward(self, params_env, params_reac):
        T_min, T_max, alpha, beta, gamma = params_reac.values()
        T_gas = params_env["T_gas"]
        cond = (T_gas >= T_min) & (T_gas < T_max)
        cond = cond.type(T_gas.dtype)
        rate = alpha*beta*(0.62 + 0.4767*gamma*(300./T_gas).sqrt())*cond
        return rate
    

class Ionpol2(nn.Module):
    def forward(self, params_env, params_reac):
        T_min, T_max, alpha, beta, gamma = params_reac.values()
        T_gas = params_env["T_gas"]
        cond = (T_gas >= T_min) & (T_gas < T_max)
        cond = cond.type(T_gas.dtype)
        inv_T_300 = 300./T_gas
        rate = alpha*beta *(1 + 0.0967*gamma*inv_T_300.sqrt()+ gamma*gamma/10.526*inv_T_300)*cond
        return rate


class GasGrainReaction(nn.Module):
    def forward(self, params_env, params_reac):
        """
        Args:
            params_reac (KeyTensor): (R, 5). Reaction parameters.
            params_env (KeyTensor): (3,). Environment parameters.

        Returns:
            tensor: (B, R). Reaction rate.
        """
        T_gas = params_env["T_gas"]
        rate = params_reac["alpha"]*(T_gas/300.)**params_reac["beta"]
        return rate


def create_gas_reaction_module_1st(formula, rmat, module_env, params_reac, meta_params):
    return FormulaDictReactionRate(
        builtin_gas_reaction_formulae_1st(meta_params), formula, rmat, module_env, params_reac)


def create_gas_reaction_module_2nd(formula, rmat, module_env, params_reac, meta_params):
    return FormulaDictReactionRate(
        builtin_gas_reaction_formulae_2nd(meta_params), formula, rmat, module_env, params_reac)


def builtin_gas_reaction_formulae_1st(meta_params):
    return {
        "CR ionization": CosmicRayIonization(meta_params.rate_cr_ion),
        "photodissociation": Photodissociation(),
    }


def builtin_gas_reaction_formulae_2nd(meta_params):
    return {
        "modified Arrhenius": ModifiedArrhenius(),
        "ionpol1": Ionpol1(),
        "ionpol2": Ionpol2(),
        "gas grain": GasGrainReaction(),
    }