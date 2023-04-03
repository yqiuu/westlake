import torch
from torch import nn
from torch.nn import functional as F


class GasPhaseReaction(nn.Module):
    def __init__(self, formula_dict, formula, rmat, params_reac, params_env):
        super(GasPhaseReaction, self).__init__()

        lookup = {key: idx for idx, key in enumerate(formula_dict.keys())}
        mask = F.one_hot(torch.tensor([lookup[name] for name in formula]), len(lookup))
        mask *= rmat.rate_sign[:, None]
        self.register_buffer("mask", mask.type(torch.float32)) # (R, F)
        self.formula_list = nn.ModuleList(formula_dict.values())
        params_reac.register_buffer(self, "params_reac")
        params_env.register_buffer(self, "params_env")

    def forward(self, t_in):
        rate = torch.zeros_like(self.mask)
        for i_f, formula in enumerate(self.formula_list):
            rate[:, i_f] = formula(self.params_reac, self.params_env)
        rate = torch.sum(rate*self.mask, dim=-1)
        return rate


class GasPhaseReaction1st(GasPhaseReaction):
    def __init__(self, formula, rmat, params_reac, params_env, meta_params):
        formula_dict = {
            "CR ionization": CosmicRayIonization(meta_params.rate_cr_ion),
            "photodissociation": Photodissociation(),
        }
        super(GasPhaseReaction1st, self).__init__(
            formula_dict, formula, rmat, params_reac, params_env)


class GasPhaseReaction2nd(GasPhaseReaction):
    def __init__(self, formula, rmat, params_reac, params_env, meta_params):
        formula_dict = {
            "modified Arrhenius": ModifiedArrhenius(),
        }
        super(GasPhaseReaction2nd, self).__init__(
            formula_dict, formula, rmat, params_reac, params_env)


class CosmicRayIonization(nn.Module):
    """Cosmic-ray ionization.

    Args:
        rate_cr_ion (float): H2 cosmic-ray ionization rate [s^-1].
    """
    def __init__(self, rate_cr_ion):
        super(CosmicRayIonization, self).__init__()
        self.register_buffer("rate_cr_ion", torch.tensor(rate_cr_ion, dtype=torch.float32))

    def forward(self, params_reac, params_env):
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
    def forward(self, params_reac, params_env):
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

        rate = alpha*torch.exp(-gamma*Av)
        return rate


class ModifiedArrhenius(nn.Module):
    def forward(self, params_reac, params_env):
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

        # TODO: Check how to compute the rate if the temperature is beyond the
        # range.
        t_width = T_max - T_min
        T_gas = (T_gas - T_min)/t_width
        T_gas = F.hardtanh(T_gas, min_val=0.)
        T_gas = T_min + (T_max - T_min)*T_gas

        rate = alpha*(T_gas/300.)**beta*torch.exp(-gamma/T_gas)
        return rate


class GasGrainReaction(nn.Module):
    def forward(self, params, temp):
        """
        Args:
            params (tensor): (R, 5). Parameters.

                - Minimum reaction temperature.
                - Maximum reaction temperature.
                - Alpha.
                - Beta.
                - Gamma.

            temp (tensor): (B,). Temperature [K].

        Returns:
            tensor: (B, R). Reaction rate.
        """
        temp = temp[:, None]
        T_min, T_max, alpha, beta, _ = params.T
        rate = alpha*(temp/300.)**beta
        return rate
