import torch
from torch import nn
from torch.func import jacrev

from .utils import TensorDict
from .assemblers import Assembler


class ConstantRateTerm(nn.Module):
    def __init__(self, rmat_1st, rmod_1st, rmat_2nd, rmod_2nd, den_norm=None):
        super(ConstantRateTerm, self).__init__()
        self.register_reactions("1st", rmat_1st, rmod_1st)
        self.register_reactions("2nd", rmat_2nd, rmod_2nd)
        self.inds_id_1st = rmat_1st.inds_id_uni
        self.inds_id_2nd = rmat_2nd.inds_id_uni
        if den_norm is None:
            den_norm = torch.ones(1)
        else:
            den_norm = torch.tensor(float(den_norm))
        self.register_buffer("den_norm", den_norm)

    def register_reactions(self, postfix, rmat, rmod):
        setattr(self, f"asm_{postfix}", Assembler(rmat))
        setattr(self, f"rmod_{postfix}", rmod)

    def forward(self, t_in, y_in):
        rates_1st, rates_2nd = self.compute_rates()
        return self.asm_1st(y_in, rates_1st, self.den_norm) \
            + self.asm_2nd(y_in, rates_2nd, self.den_norm)

    def jacobian(self, t_in, y_in):
        rates_1st, rates_2nd = self.compute_rates()
        return self.asm_1st.jacobain(y_in, rates_1st, self.den_norm) \
            + self.asm_2nd.jacobain(y_in, rates_2nd, self.den_norm)

    def compute_rates(self):
        rates_1st = self.rmod_1st()
        rates_2nd = self.rmod_2nd()
        return rates_1st, rates_2nd


class TwoPhaseTerm(nn.Module):
    def __init__(self, rmat_1st, rmod_1st, rmat_2nd, rmod_2nd, module_med=None):
        super(TwoPhaseTerm, self).__init__()
        if module_med is None \
            or isinstance(module_med, TensorDict) or isinstance(module_med, nn.Module):
            self.module_med = module_med
        else:
            raise ValueError("Unknown 'module_med'.")

        self.register_reactions("1st", rmat_1st, rmod_1st)
        self.register_reactions("2nd", rmat_2nd, rmod_2nd)
        self.inds_id_1st = rmat_1st.inds_id_uni
        self.inds_id_2nd = rmat_2nd.inds_id_uni

    def register_reactions(self, postfix, rmat, rmod):
        setattr(self, f"asm_{postfix}", Assembler(rmat))
        setattr(self, f"rmod_{postfix}", rmod)

    def forward(self, t_in, y_in, **params_extra):
        rates_1st, rates_2nd, den_norm = self.compute_rates(t_in, y_in, **params_extra)
        return self.asm_1st(y_in, rates_1st, den_norm) \
            + self.asm_2nd(y_in, rates_2nd, den_norm)

    def jacobian(self, t_in, y_in, **params_extra):
        rates_1st, rates_2nd, den_norm = self.compute_rates(t_in, y_in, **params_extra)
        return self.asm_1st.jacobain(y_in, rates_1st, den_norm) \
            + self.asm_2nd.jacobain(y_in, rates_2nd, den_norm)

    def compute_rates(self, t_in, y_in, **params_extra):
        if self.module_med is None:
            params_med = None
            den_norm = None
        else:
            params_med = self.module_med(t_in, **params_extra)
            den_norm = params_med['den_gas']
        rates_1st = self.rmod_1st(t_in, params_med)
        rates_2nd = self.rmod_2nd(t_in, params_med)
        return rates_1st, rates_2nd, den_norm

    def reproduce_rate_coeffs(self, t_in=None, y_in=None):
        if t_in is None:
            t_in = torch.tensor([0.])

        inds_id_1st = self.inds_id_1st
        inds_id_2nd = self.inds_id_2nd
        n_reac = len(inds_id_1st) + len(inds_id_2nd)
        rates = torch.zeros([len(t_in), n_reac])
        with torch.no_grad():
            params_med = self.module_med(t_in)
            rates[:, inds_id_1st] = self.rmod_1st.compute_rates_reac(t_in, params_med)
            rates[:, inds_id_2nd] = self.rmod_2nd.compute_rates_reac(t_in, params_med)
        rates = rates.T.squeeze()
        return rates


class ThreePhaseTerm(nn.Module):
    def __init__(self, rmod_smt, rmat_smt,
                 rmod_1st, rmat_1st, rmat_1st_surf_gain, rmat_1st_surf_loss, rmat_1st_other,
                 rmod_2nd, rmat_2nd, rmat_2nd_surf_gain, rmat_2nd_surf_loss, rmat_2nd_other,
                 inds_surf, inds_mant, module_med=None):
        super().__init__()
        if module_med is None \
            or isinstance(module_med, TensorDict) or isinstance(module_med, nn.Module):
            self.module_med = module_med
        else:
            raise ValueError("Unknown 'module_med'.")
        #
        self.rmod_smt = rmod_smt
        self.asm_smt = Assembler(rmat_smt)
        # First order reactions
        self.rmod_1st = rmod_1st
        self.asm_1st = Assembler(rmat_1st)
        self.asm_1st_surf_gain = Assembler(rmat_1st_surf_gain)
        self.asm_1st_surf_loss = Assembler(rmat_1st_surf_loss)
        self.asm_1st_other = Assembler(rmat_1st_other)
        # Second order reactions
        self.rmod_2nd = rmod_2nd
        self.asm_2nd = Assembler(rmat_2nd)
        self.asm_2nd_surf_gain = Assembler(rmat_2nd_surf_gain)
        self.asm_2nd_surf_loss = Assembler(rmat_2nd_surf_loss)
        self.asm_2nd_other = Assembler(rmat_2nd_other)
        #
        self.register_buffer("inds_surf", torch.tensor(inds_surf))
        self.register_buffer("inds_mant", torch.tensor(inds_mant))
        #
        self.inds_id_smt = rmat_smt.inds_id_uni
        self.inds_id_1st = rmat_1st.inds_id_uni
        self.inds_id_2nd = rmat_2nd.inds_id_uni

    def forward(self, t_in, y_in, **params_extra):
        rates_smt, rates_1st, rates_2nd, den_norm = self.compute_rates(t_in, y_in)
        return self.asm_smt(y_in, rates_smt, den_norm) \
            + self.asm_1st(y_in, rates_1st, den_norm) \
            + self.asm_2nd(y_in, rates_2nd, den_norm)

    def jacobian(self, t_in, y_in, **params_extra):
        return jacrev(self, argnums=1)(t_in, y_in)

    def compute_rates(self, t_in, y_in, **params_extra):
        if self.module_med is None:
            params_med = None
            den_norm = None
        else:
            params_med = self.module_med(t_in, **params_extra)
            den_norm = params_med['den_gas']
        rates_1st = self.rmod_1st(t_in, params_med)
        rates_2nd = self.rmod_2nd(t_in, params_med)

        y_in = torch.atleast_2d(y_in)
        rates_1st = torch.atleast_2d(rates_1st)
        rates_2nd = torch.atleast_2d(rates_2nd)

        dy_1st_gain = self.asm_1st_surf_gain(y_in, rates_1st, den_norm)[:, self.inds_surf]
        dy_2nd_gain = self.asm_2nd_surf_gain(y_in, rates_2nd, den_norm)[:, self.inds_surf]
        dy_surf_gain = torch.sum(dy_1st_gain + dy_2nd_gain, dim=-1, keepdim=True)

        dy_1st_loss = self.asm_1st_surf_loss(y_in, rates_1st, den_norm)[:, self.inds_surf]
        dy_2nd_loss = self.asm_2nd_surf_loss(y_in, rates_2nd, den_norm)[:, self.inds_surf]
        dy_surf_loss = -torch.sum(dy_1st_loss + dy_2nd_loss, dim=-1, keepdim=True)

        rates_smt = self.rmod_smt(
            params_med, y_in, self.inds_surf, self.inds_mant, dy_surf_gain, dy_surf_loss,
        )
        return rates_smt, rates_1st, rates_2nd, den_norm

    def reproduce_rate_coeffs(self, t_in, y_in):
        inds_id_smt = self.inds_id_smt
        inds_id_1st = self.inds_id_1st
        inds_id_2nd = self.inds_id_2nd
        n_reac = len(inds_id_smt) + len(inds_id_1st) + len(inds_id_2nd)
        rates = torch.zeros([len(t_in), n_reac])
        with torch.no_grad():
            rates_smt, rates_1st, rates_2nd, _ = self.compute_rates(t_in, y_in)
            rates[:, inds_id_smt] = rates_smt
            rates[:, inds_id_1st] = rates_1st[:, self.rmod_1st.inds_reac]
            rates[:, inds_id_2nd] = rates_2nd[:, self.rmod_2nd.inds_reac]
        rates = rates.T.squeeze()
        return rates