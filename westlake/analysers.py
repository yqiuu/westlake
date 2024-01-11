import numpy as np


class MainReactionTracer:
    len_reac = 40

    def __init__(self, df_reac, df_spec, time, ab, den_gas, rates, is_coeff, special_species):
        df_reac = df_reac[["key", "reactant_1", "reactant_2", "products", "formula"]].copy()
        df_reac["products"] = df_reac["products"].map(lambda name: tuple(name.split(";")))
        if is_coeff:
            rates = self.compute_reaction_rates(
                df_reac, df_spec, ab, den_gas, rates, special_species
            )
        self._df_reac = df_reac
        self._rates = rates
        self._time = time

    @classmethod
    def from_result(cls, res, df_reac, df_spec, config):
        if res.coeffs is None:
            raise ValueError("Rate coefficients are not saved. Set save_rate_coeffs=True.")
        return cls(
            df_reac, df_spec, res.time, res.ab, res.den_gas, res.coeffs,
            is_coeff=True, special_species=config.special_species
        )

    def compute_reaction_rates(self, df_reac, df_spec, ab, den_gas, coeffs, special_species):
        inds_r = df_spec.index.get_indexer(df_reac["reactant_1"])
        rates = coeffs*ab[inds_r]*den_gas

        cond = df_reac["reactant_2"].map(lambda name: name not in special_species and name != "")
        inds_r = df_spec.index.get_indexer(df_reac["reactant_2"][cond])
        rates[cond] = rates[cond]*ab[inds_r]*den_gas
        return rates

    def trace_instant(self, specie, t_form, percent_cut=1., rate_cut=0., quiet=False):
        cond_prod, cond_dest = self._select_prod_dest(specie)

        idx_t = np.argmin(np.abs(t_form - self._time))
        df_prod = self._trace(cond_prod, percent_cut, rate_cut, idx_t)
        df_dest = self._trace(cond_dest, percent_cut, rate_cut, idx_t)

        if not quiet:
            print("idx_t = {}, t = {:.3e}".format(idx_t, self._time[idx_t]))
            print("\nProduction")
            self._print_reactions(df_prod)
            print("\nDestruction")
            self._print_reactions(df_dest)

        return df_prod, df_dest

    def trace_period(self, specie, t_start, t_end, percent_cut=1., rate_cut=0., quiet=False):
        cond_prod, cond_dest = self._select_prod_dest(specie)

        idx_start = np.argmin(np.abs(t_start - self._time))
        idx_end = np.argmin(np.abs(t_end - self._time))
        if idx_end <= idx_start:
            raise ValueError("'t_end' should be greater than 't_start'.")

        df_prod = self._trace(cond_prod, percent_cut, rate_cut, idx_start, idx_end)
        df_dest = self._trace(cond_dest, percent_cut, rate_cut, idx_start, idx_end)

        if not quiet:
            print("idx_start = {}, t = {:.3e}".format(idx_start, self._time[idx_start]))
            print("idx_end = {}, t = {:.3e}".format(idx_end, self._time[idx_end]))
            print("\nProduction")
            self._print_reactions(df_prod)
            print("\nDestruction")
            self._print_reactions(df_dest)

        return df_prod, df_dest

    def _select_prod_dest(self, specie):
        df_reac = self._df_reac
        cond_prod = df_reac["products"].map(lambda specs: specie in specs)
        cond_dest = (df_reac["reactant_1"] == specie) \
            | (df_reac["reactant_2"] == specie)
        if np.count_nonzero(cond_prod) + np.count_nonzero(cond_dest) == 0:
            raise ValueError(f"{specie} does not exist.")
        return cond_prod, cond_dest

    def _trace(self, cond, percent_cut, rate_cut, idx_start, idx_end=None):
        df_sub = self._df_reac.loc[cond]
        rates = self._rates[cond]

        if idx_end is None:
            rates = rates[:, idx_start]
        else:
            time = self._time
            rates = np.trapz(rates[:, idx_start : idx_end+1], time[idx_start : idx_end+1], axis=-1)
            rates = rates/(time[idx_end] - time[idx_start])
        norm = rates.sum(axis=0, keepdims=True)
        norm[norm == 0.] = 1.
        percents = rates/norm*100.

        cond = (percents >= percent_cut) & (rates >= rate_cut)
        rates = rates[cond]
        percents = percents[cond]
        df_sub = df_sub[cond].copy()

        inds = np.argsort(percents)[::-1]
        rates = rates[inds]
        percents = percents[inds]
        df_sub = df_sub.iloc[inds]

        df_ret = df_sub[["reactant_1", "reactant_2", "products", "formula"]].copy()
        df_ret["products"] = df_ret["products"].map(lambda prods: ";".join(prods))
        df_ret["rate"] = rates
        df_ret["percent"] = percents
        return df_ret

    def _print_reactions(self, df_reac):
        for idx, reac_1, reac_2, prods, rate, percent in zip(
            df_reac.index, df_reac["reactant_1"], df_reac["reactant_2"],
            df_reac["products"], df_reac["rate"], df_reac["percent"]
        ):
            if reac_2 == "":
                reac = f"{reac_1} -> "
            else:
                reac = f"{reac_1} + {reac_2} -> "
            reac += prods.replace(";", " + ")
            line = "{:5d}   ".format(idx) + reac \
            + " "*(self.len_reac - len(reac)) + "{:.5e}   {:.1f}%".format(rate, percent)
            print(line)