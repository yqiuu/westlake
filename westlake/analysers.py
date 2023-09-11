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


    def trace(self, specie, t_form, percent_cut=1, rate_cut=0.):
        cond = self._df_reac["products"].map(lambda specs: specie in specs)
        if np.count_nonzero(cond) == 0:
            raise ValueError(f"{specie} does not exist.")

        df_sub = self._df_reac.loc[cond]
        rates = self._rates[cond]
        percents = rates/rates.sum(axis=0, keepdims=True)*100.

        idx_t = np.argmin(np.abs(t_form - self._time))
        rates = rates[:, idx_t]
        percents = percents[:, idx_t]

        cond = (percents > percent_cut) & (rates > rate_cut)
        rates = rates[cond]
        percents = percents[cond]
        df_sub = df_sub[cond].copy()

        inds = np.argsort(percents)[::-1]
        rates = rates[inds]
        percents = percents[inds]
        df_sub = df_sub.iloc[inds]

        print("idx_t = {}, t = {:.3e}".format(idx_t, self._time[idx_t]))
        for idx, reac, rate, percent in zip(df_sub.index, df_sub["key"], rates, percents):
            reac = reac.replace(";", " + ")
            reac = reac.replace(">", " > ")
            line = "{:5d}   ".format(idx) + reac \
                + " "*(40 - len(reac)) + "{:.5e}   {:.1f}%".format(rate, percent)
            print(line)

        return df_sub, rates, percents