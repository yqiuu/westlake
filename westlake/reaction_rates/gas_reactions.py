import torch
from torch import nn
from torch.nn import functional as F

from .reaction_rates import ReactionRate


def builtin_gas_reactions(config):
    return {
        "CR dissociation": CRDissociation(config),
        "CRP dissociation": CRDissociation(config),
        "photodissociation": Photodissociation(config),
        "modified Arrhenius": ModifiedArrhenius(),
        "ionpol1": Ionpol1(),
        "ionpol2": Ionpol2(),
        "gas grain": GasGrainReaction(),
        "surface H2 formation": SurfaceH2Formation(config),
        "surface H accretion": SurfaceHAccretion(config),
    }


class CRDissociation(ReactionRate):
    def __init__(self, config):
        super().__init__(["alpha"])
        self.register_buffer("xi_cr_xr", torch.tensor(config.rate_cr_ion + config.rate_x_ion))

    def forward(self, params_med, params_reac, **params_extra):
        """
        Args:
            params_reac (KeyTensor): (R, 5). Reaction parameters.
            params_med (KeyTensor): (3,). Environment parameters.

        Returns:
            tensor: (R,). Reaction rate.
        """
        return params_reac["alpha"]*self.xi_cr_xr


class Photodissociation(ReactionRate):
    def __init__(self, config):
        super().__init__(["alpha", "gamma"])
        self.register_buffer("uv_flux", torch.tensor(config.uv_flux))

    def forward(self, params_med, params_reac, **params_extra):
        """
        Args:
            params_reac (KeyTensor): (R, 5). Reaction parameters.
            params_med (KeyTensor): (3,). Environment parameters.

        Returns:
            tensor: (R,). Reaction rate.
        """
        return params_reac["alpha"]*torch.exp(-params_reac["gamma"]*params_med["Av"])*self.uv_flux


class ModifiedArrhenius(ReactionRate):
    def __init__(self):
        super().__init__([
            "alpha", "beta", "gamma",
            "T_min", "T_max", "is_leftmost", "is_rightmost"
        ])

    def forward(self, params_med, params_reac, **params_extra):
        """
        Args:
            params_reac (KeyTensor): (R, 5). Reaction parameters.
            params_med (KeyTensor): (3,). Environment parameters.

        Returns:
            tensor: (B, R). Reaction rate.
        """
        alpha = params_reac["alpha"]
        beta = params_reac["beta"]
        gamma = params_reac["gamma"]
        T_gas, mask_T = clamp_gas_temperature(params_med, params_reac)
        rate = alpha*(T_gas/300.)**beta*torch.exp(-gamma/T_gas)*mask_T
        return rate


class Ionpol1(ReactionRate):
    def __init__(self):
        super().__init__(["alpha", "beta", "gamma"])

    def forward(self, params_med, params_reac, **params_extra):
        alpha = params_reac["alpha"]
        beta = params_reac["beta"]
        gamma = params_reac["gamma"]
        T_gas, mask_T = clamp_gas_temperature(params_med, params_reac)
        rate = alpha*beta*(0.62 + 0.4767*gamma*(300./T_gas).sqrt())*mask_T
        return rate


class Ionpol2(ReactionRate):
    def __init__(self):
        super().__init__(["alpha", "beta", "gamma"])

    def forward(self, params_med, params_reac, **params_extra):
        alpha = params_reac["alpha"]
        beta = params_reac["beta"]
        gamma = params_reac["gamma"]
        T_gas, mask_T = clamp_gas_temperature(params_med, params_reac)
        inv_T_300 = 300./T_gas
        rate = alpha*beta *(1 + 0.0967*gamma*inv_T_300.sqrt()+ gamma*gamma/10.526*inv_T_300)*mask_T
        return rate


class GasGrainReaction(ReactionRate):
    def __init__(self):
        super().__init__(["alpha", "beta"])

    def forward(self, params_med, params_reac, **params_extra):
        """
        Args:
            params_reac (KeyTensor): (R, 5). Reaction parameters.
            params_med (KeyTensor): (3,). Environment parameters.

        Returns:
            tensor: (B, R). Reaction rate.
        """
        T_gas = params_med["T_gas"]
        rate = params_reac["alpha"]*(T_gas/300.)**params_reac["beta"]
        return rate


class SurfaceHAccretion(ReactionRate):
    def __init__(self, config):
        super().__init__(["alpha", "beta"])
        self.register_buffer("dtg_num_ratio", torch.tensor(config.dtg_num_ratio))

    def forward(self, params_med, params_reac, **params_extra):
        return params_reac["alpha"]*(params_med["T_gas"]/300)**params_reac["beta"] \
            *self.dtg_num_ratio*params_med["den_gas"]


class SurfaceH2Formation(ReactionRate):
    def __init__(self, config):
        super().__init__(["alpha"])
        self.register_buffer("inv_dtg_num_ratio", torch.tensor(1./config.dtg_num_ratio))

    def forward(self, params_med, params_reac, **params_extra):
        return 1.186e7*params_reac["alpha"]*torch.exp(-225./params_med["T_gas"]) \
            *self.inv_dtg_num_ratio/params_med["den_gas"]


def clamp_gas_temperature(params_med, params_reac):
    is_leftmost = params_reac["is_leftmost"]
    is_rightmost = params_reac["is_rightmost"]
    T_min = params_reac["T_min"] # (R,) Number of reactions
    T_max = params_reac["T_max"] # (R,)
    T_gas = params_med["T_gas"] # (B, 1)
    cond_ge = T_gas >= T_min
    cond_lt = T_gas < T_max
    factor_0 = (cond_ge | is_leftmost) & (cond_lt | is_rightmost)
    factor_0 = factor_0.type(T_gas.dtype)
    width = 1.
    factor_1 = F.hardtanh((T_gas - T_min)/width + 1, min_val=0.) \
        * F.hardtanh((T_max - T_gas)/width + 1, min_val=0.)
    factor_T = torch.maximum(factor_0, factor_1)

    T_gas = T_gas.repeat(1, T_min.shape[0]) # (B, R)
    T_gas = torch.where(cond_ge, T_gas, T_min)
    T_gas = torch.where(cond_lt, T_gas, T_max)
    return T_gas, factor_T
