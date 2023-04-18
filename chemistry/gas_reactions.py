import torch
from torch import nn
from torch.nn import functional as F

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
        rate = params_reac.get("alpha")*self.rate_cr_ion
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
        alpha = params_reac.get("alpha")
        gamma = params_reac.get("gamma")
        Av = params_env.get("Av")
        if len(Av) != 1:
            Av = Av[:, None]
        rate = alpha*torch.exp(-gamma*Av)
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
        T_min, T_max, alpha, beta, gamma \
            = params_reac.get(("T_min", "T_max", "alpha", "beta", "gamma")).T
        T_gas = params_env.get("T_gas")
        if len(T_gas) != 1:
            T_gas = T_gas[:, None]

        # TODO: Check how to compute the rate if the temperature is beyond the
        # range.
        T_width = T_max - T_min
        T_gas = (T_gas - T_min)/T_width
        T_gas = F.hardtanh(T_gas, min_val=0.)
        T_gas = T_min + (T_max - T_min)*T_gas

        rate = alpha*(T_gas/300.)**beta*torch.exp(-gamma/T_gas)
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
        alpha = params_reac.get("alpha")
        beta = params_reac.get("beta")
        T_gas = params_env.get("T_gas")
        rate = alpha*(T_gas/300.)**beta
        return rate


def create_gas_reactions_1st(formula, rmat, module_env, params_reac, meta_params):
    formula_dict = {
        "CR ionization": CosmicRayIonization(meta_params.rate_cr_ion),
        "photodissociation": Photodissociation(),
    }
    return FormulaDictReactionRate(formula_dict, formula, rmat, module_env, params_reac)


def create_gas_reactions_2nd(formula, rmat, module_env, params_reac, meta_params):
    formula_dict = {
        "modified Arrhenius": ModifiedArrhenius(),
        "gas grain": GasGrainReaction(),
    }
    return FormulaDictReactionRate(formula_dict, formula, rmat, module_env, params_reac)
