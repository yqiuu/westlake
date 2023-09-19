import numpy as np


class MainReactionTracer:
    def __init__(self, df_reac, df_spec, time, ab, rates, den_gas, is_coeff=True):
        df_reac = df_reac[["key", "reactant_1", "reactant_2", "products"]].copy()
        df_reac["products"] = df_reac["products"].map(lambda name: tuple(name.split(";")))

        if is_coeff:
            rates_coeff = rates
            inds_r = df_spec.loc[df_reac["reactant_1"], "index"]
            rates = rates_coeff*ab[inds_r]*den_gas

            cond = df_reac["reactant_2"] != ""
            inds_r = df_spec.loc[df_reac.loc[cond, "reactant_2"], "index"]
            rates[cond] = rates[cond]*ab[inds_r]*den_gas

        self._df_reac = df_reac
        self._rates = rates
        self._time = time

    def trace_instant(self, specie, t_form, percent_cut=1., rate_cut=0.):
        cond_prod, cond_dest = self._select_prod_dest(specie)

        idx_t = np.argmin(np.abs(t_form - self._time))
        print("idx_t = {}, t = {:.3e}".format(idx_t, self._time[idx_t]))

        print("Production")
        self._trace(cond_prod, percent_cut, rate_cut, idx_t)
        print("\nDestruction")
        self._trace(cond_dest, percent_cut, rate_cut, idx_t)

    def trace_period(self, specie, t_start, t_end, percent_cut=1., rate_cut=0.):
        cond_prod, cond_dest = self._select_prod_dest(specie)

        idx_start = np.argmin(np.abs(t_start - self._time))
        print("idx_start = {}, t = {:.3e}".format(idx_start, self._time[idx_start]))
        idx_end = np.argmin(np.abs(t_end - self._time))
        print("idx_end = {}, t = {:.3e}".format(idx_end, self._time[idx_end]))
        if idx_end <= idx_start:
            raise ValueError("'t_end' should be greater than 't_start'.")

        print("Production")
        self._trace(cond_prod, percent_cut, rate_cut, idx_start, idx_end)
        print("\nDestruction")
        self._trace(cond_dest, percent_cut, rate_cut, idx_start, idx_end)

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
        percents = rates/rates.sum(axis=0, keepdims=True)*100.

        cond = (percents > percent_cut) & (rates > rate_cut)
        rates = rates[cond]
        percents = percents[cond]
        df_sub = df_sub[cond].copy()

        inds = np.argsort(percents)[::-1]
        rates = rates[inds]
        percents = percents[inds]
        df_sub = df_sub.iloc[inds]

        for idx, reac, rate, percent in zip(df_sub.index, df_sub["key"], rates, percents):
            reac = reac.replace(";", " + ")
            reac = reac.replace(">", " > ")
            line = "{:5d}   ".format(idx) + reac \
                + " "*(40 - len(reac)) + "{:.5e}   {:.1f}%".format(rate, percent)
            print(line)

        return df_sub, rates, percents