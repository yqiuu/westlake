import torch
from torch import nn

from .utils import TensorDict
from .assemblers import Assembler


class ConstantRateTerm(nn.Module):
    def __init__(self, rmat_1st, rmat_2nd, rates, den_norm=None):
        super(ConstantRateTerm, self).__init__()
        self.register_reactions("1st", rmat_1st, rates)
        self.register_reactions("2nd", rmat_2nd, rates)
        if den_norm is None:
            den_norm = torch.ones(1)
        else:
            den_norm = torch.tensor(float(den_norm))
        self.register_buffer("den_norm", den_norm)

    def register_reactions(self, postfix, rmat, rates):
        rates = torch.tensor(rmat.rate_sign*rates[rmat.inds_k], dtype=torch.get_default_dtype())
        self.register_buffer(f"rates_{postfix}", rates)
        rmat.rate_sign = None
        setattr(self, f"asm_{postfix}", Assembler(rmat))

    def forward(self, t_in, y_in):
        return self.asm_1st(y_in, self.rates_1st, self.den_norm) \
            + self.asm_2nd(y_in, self.rates_2nd, self.den_norm)

    def jacobian(self, t_in, y_in):
        return self.asm_1st.jacobain(y_in, self.rates_1st, self.den_norm) \
            + self.asm_2nd.jacobain(y_in, self.rates_2nd, self.den_norm)


class TwoPhaseTerm(nn.Module):
    def __init__(self, rmod, rmod_ex, module_var, rmat_1st, rmat_2nd, module_med):
        super(TwoPhaseTerm, self).__init__()
        if module_med is None \
            or isinstance(module_med, TensorDict) or isinstance(module_med, nn.Module):
            self.module_med = module_med
        else:
            raise ValueError("Unknown 'module_med'.")
        self.rmod = rmod
        self.rmod_ex = rmod_ex
        self.module_var = module_var
        self.asm_1st = Assembler(rmat_1st)
        self.asm_2nd = Assembler(rmat_2nd)
        self.inds_id_1st = rmat_1st.inds_id_uni
        self.inds_id_2nd = rmat_2nd.inds_id_uni

    def forward(self, t_in, y_in, **params_extra):
        coeffs, den_norm = self.compute_rate_coeffs(t_in, y_in, **params_extra)
        return self.asm_1st(y_in, coeffs, den_norm) \
            + self.asm_2nd(y_in, coeffs, den_norm)

    def jacobian(self, t_in, y_in, **params_extra):
        coeffs, den_norm = self.compute_rate_coeffs(t_in, y_in, **params_extra)
        return self.asm_1st.jacobain(y_in, coeffs, den_norm) \
            + self.asm_2nd.jacobain(y_in, coeffs, den_norm)

    def compute_rate_coeffs(self, t_in, y_in):
        if self.module_med is None:
            params_med = None
            den_norm = None
        else:
            params_med = self.module_med(t_in)
            den_norm = params_med['den_gas']

        inds_id_1st = self.inds_id_1st
        inds_id_2nd = self.inds_id_2nd
        n_reac = len(inds_id_1st) + len(inds_id_2nd)
        coeffs = torch.zeros([len(t_in), n_reac], device=t_in.device)
        self.rmod.assign_rate_coeffs(coeffs, params_med)
        if self.module_var is None:
            params_extra = None
        else:
            params_extra = self.module_var(coeffs, params_med, y_in)
        if self.rmod_ex is not None:
            self.rmod_ex.assign_rate_coeffs(coeffs, params_med, y_in, params_extra)
        return coeffs, den_norm

    def reproduce_rate_coeffs(self, t_in=None, y_in=None):
        if t_in is None:
            t_in = torch.tensor([0.])
        coeffs, _ = self.compute_rate_coeffs(t_in, y_in)
        coeffs = coeffs.T.squeeze()
        return coeffs


class ThreePhaseTerm(nn.Module):
    def __init__(self, rmod, rmod_ex, module_var, rmod_smt,
                 rmat_1st, rmat_1st_surf_gain, rmat_1st_surf_loss,
                 rmat_2nd, rmat_2nd_surf_gain, rmat_2nd_surf_loss,
                 inds_surf, inds_mant, module_med, config):
        super().__init__()
        if module_med is None \
            or isinstance(module_med, TensorDict) or isinstance(module_med, nn.Module):
            self.module_med = module_med
        else:
            raise ValueError("Unknown 'module_med'.")
        #
        self.rmod = rmod
        self.rmod_ex = rmod_ex
        self.module_var = module_var
        self.rmod_smt = rmod_smt
        #
        self.asm_1st = Assembler(rmat_1st)
        self.asm_1st_surf_gain = Assembler(rmat_1st_surf_gain)
        self.asm_1st_surf_loss = Assembler(rmat_1st_surf_loss)
        #
        self.asm_2nd = Assembler(rmat_2nd)
        self.asm_2nd_surf_gain = Assembler(rmat_2nd_surf_gain)
        self.asm_2nd_surf_loss = Assembler(rmat_2nd_surf_loss)
        #
        self.inds_id_1st = rmat_1st.inds_id_uni
        self.inds_id_2nd = rmat_2nd.inds_id_uni
        #
        self.register_buffer("inds_surf", torch.tensor(inds_surf))
        self.register_buffer("inds_mant", torch.tensor(inds_mant))
        self.register_buffer("layer_factor", torch.tensor(config.layer_factor))
        self.register_buffer("num_active_layers", torch.tensor(config.num_active_layers))

    def forward(self, t_in, y_in, **params_extra):
        coeffs, den_norm = self.compute_rate_coeffs(t_in, y_in, **params_extra)
        return self.asm_1st(y_in, coeffs, den_norm) \
            + self.asm_2nd(y_in, coeffs, den_norm)

    def jacobian(self, t_in, y_in, **params_extra):
        coeffs, den_norm = self.compute_rate_coeffs(t_in, y_in, **params_extra)
        return self.asm_1st.jacobain(y_in, coeffs, den_norm) \
            + self.asm_2nd.jacobain(y_in, coeffs, den_norm)

    def compute_rate_coeffs(self, t_in, y_in, **params_extra):
        params_med = self.module_med(t_in, **params_extra)
        den_norm = params_med['den_gas']
        inds_id_1st = self.inds_id_1st
        inds_id_2nd = self.inds_id_2nd
        n_reac = len(inds_id_1st) + len(inds_id_2nd)
        coeffs = torch.zeros([len(t_in), n_reac], device=t_in.device)

        y_in = torch.atleast_2d(y_in)
        y_surf = y_in[:, self.inds_surf].sum(dim=-1, keepdim=True)
        y_mant = y_in[:, self.inds_mant].sum(dim=-1, keepdim=True)
        n_layer_surf = self.layer_factor*y_surf
        n_layer_mant = self.layer_factor*y_mant
        decay_factor = self.num_active_layers/(n_layer_surf + n_layer_mant)
        deacy_factor = decay_factor.clamp_max(1.)
        params_extra = {
            "n_layer_surf": n_layer_surf,
            "n_layer_mant": n_layer_mant,
            "decay_factor": deacy_factor
        }

        #
        self.rmod.assign_rate_coeffs(coeffs, params_med)
        if self.module_var is not None:
            params_extra = self.module_var(coeffs, params_med, y_in, params_extra)
        if self.rmod_ex is not None:
            self.rmod_ex.assign_rate_coeffs(coeffs, params_med, y_in, params_extra)

        dy_1st_gain = self.asm_1st_surf_gain(y_in, coeffs, den_norm)[:, self.inds_surf]
        dy_2nd_gain = self.asm_2nd_surf_gain(y_in, coeffs, den_norm)[:, self.inds_surf]
        dy_surf_gain = torch.sum(dy_1st_gain + dy_2nd_gain, dim=-1, keepdim=True)

        dy_1st_loss = self.asm_1st_surf_loss(y_in, coeffs, den_norm)[:, self.inds_surf]
        dy_2nd_loss = self.asm_2nd_surf_loss(y_in, coeffs, den_norm)[:, self.inds_surf]

        dy_surf_loss = -torch.sum(dy_1st_loss + dy_2nd_loss, dim=-1, keepdim=True)

        self.rmod_smt.assign_rate_coeffs(
            coeffs, params_med, y_in, self.inds_mant,
            y_surf, y_mant, dy_surf_gain, dy_surf_loss
        )
        return coeffs, den_norm

    def reproduce_rate_coeffs(self, t_in, y_in):
        coeffs, _ = self.compute_rate_coeffs(t_in, y_in)
        coeffs = coeffs.T.squeeze()
        return coeffs


class VariableModule(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self._name_list = []

    def add_variable(self, name, module):
        self._name_list.append(name)
        self.append(module)

    def forward(self, coeffs, params_med, y_in, var_dict=None):
        if var_dict is None:
            var_dict = {}
        for name, module in zip(self._name_list, self):
            if isinstance(name, str):
                var_dict[name] = module(coeffs, params_med, y_in, var_dict)
            else:
                variables = module(coeffs, params_med, y_in, var_dict)
                for nm, var in zip(name, variables):
                    var_dict[nm] = var
        return var_dict