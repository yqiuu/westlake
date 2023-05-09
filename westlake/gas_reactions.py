import torch
from torch import nn
from torch.nn import functional as F


def builtin_gas_reactions_1st(meta_params):
    return {
        "CR ionization": CosmicRayIonization(meta_params.rate_cr_ion),
        "photodissociation": Photodissociation(),
    }


def builtin_gas_reactions_2nd(meta_params):
    return {
        "modified Arrhenius": ModifiedArrhenius(),
        "ionpol1": Ionpol1(),
        "ionpol2": Ionpol2(),
        "gas grain": GasGrainReaction(),
    }


class CosmicRayIonization(nn.Module):
    """Cosmic-ray ionization.

    Args:
        rate_cr_ion (float): H2 cosmic-ray ionization rate [s^-1].
    """
    def __init__(self, rate_cr_ion):
        super(CosmicRayIonization, self).__init__()
        self.register_buffer("rate_cr_ion", torch.tensor(rate_cr_ion))

    def forward(self, params_env, params_reac, **params_extra):
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
    def forward(self, params_env, params_reac, **params_extra):
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
    def forward(self, params_env, params_reac, **params_extra):
        """
        Args:
            params_reac (KeyTensor): (R, 5). Reaction parameters.
            params_env (KeyTensor): (3,). Environment parameters.

        Returns:
            tensor: (B, R). Reaction rate.
        """
        alpha = params_reac["alpha"]
        beta = params_reac["beta"]
        gamma = params_reac["gamma"]
        T_gas, mask_T = clamp_gas_temperature(params_env, params_reac)
        rate = alpha*(T_gas/300.)**beta*torch.exp(-gamma/T_gas)*mask_T
        return rate


class Ionpol1(nn.Module):
    def forward(self, params_env, params_reac, **params_extra):
        alpha = params_reac["alpha"]
        beta = params_reac["beta"]
        gamma = params_reac["gamma"]
        T_gas, mask_T = clamp_gas_temperature(params_env, params_reac)
        rate = alpha*beta*(0.62 + 0.4767*gamma*(300./T_gas).sqrt())*mask_T
        return rate


class Ionpol2(nn.Module):
    def forward(self, params_env, params_reac, **params_extra):
        alpha = params_reac["alpha"]
        beta = params_reac["beta"]
        gamma = params_reac["gamma"]
        T_gas, mask_T = clamp_gas_temperature(params_env, params_reac)
        inv_T_300 = 300./T_gas
        rate = alpha*beta *(1 + 0.0967*gamma*inv_T_300.sqrt()+ gamma*gamma/10.526*inv_T_300)*mask_T
        return rate


class GasGrainReaction(nn.Module):
    def forward(self, params_env, params_reac, **params_extra):
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


def clamp_gas_temperature(params_env, params_reac):
    is_unique = params_reac["is_unique"]
    T_min = params_reac["T_min"]
    T_max = params_reac["T_max"]
    T_gas = params_env["T_gas"]
    cond_ge = T_gas >= T_min
    cond_lt = T_gas < T_max
    mask_T = cond_ge & cond_lt | is_unique
    mask_T = mask_T.type(T_gas.dtype)
    # TODO: Check the shape of T_gas
    T_gas = T_gas.repeat(is_unique.shape[0])
    T_gas = torch.where(cond_ge, T_gas, T_min)
    T_gas = torch.where(cond_lt, T_gas, T_max)
    return T_gas, mask_T