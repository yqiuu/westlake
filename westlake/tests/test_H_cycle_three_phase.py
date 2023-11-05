import os
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import torch
import westlake

from utils import get_abs_fname


torch.set_default_dtype(torch.float64)
FILE_NAME = "H_cycle_three_phase.pickle"
RTOL = 1e-5
ATOL = 1e-20


def test_H_cycle():
    res_new = solve_H_cycle()
    res_fid = pickle.load(open(get_abs_fname(FILE_NAME), "rb"))
    assert_allclose(res_fid.y, res_new.y, rtol=RTOL, atol=ATOL)


def solve_H_cycle():
    fname = get_fname("H_cycle_three_phase.h5")
    df_reac = pd.read_hdf(fname, key="reactions")
    df_spec = westlake.prepare_specie_table(df_reac, pd.read_hdf(fname, key="species"))
    df_surf = pd.read_hdf(fname, key="surface_parameters")
    df_barr = pd.DataFrame([230.], index=["JH"], columns=["E_barr"])

    ab_0_dict = {'H2': .5,}
    t_start = 0.
    t_end = 1e5
    meta_params = westlake.MetaParameters(
        model="three phase", atol=ATOL, ab_0_min=1e-40,
        t_start=t_start, t_end=t_end,
    )
    medium = westlake.StaticMedium({'Av': 10., "den_gas": 1e4, "T_gas": 17., "T_dust": 17.})
    reaction_term = westlake.create_astrochem_model(
        df_reac, df_spec, df_surf, meta_params, medium, df_barr=df_barr
    )

    n_eval = 100
    t_eval = np.linspace(t_start, t_end, n_eval)
    res = westlake.solve_rate_equation_astrochem(
        reaction_term, ab_0_dict, df_spec, meta_params, t_eval=t_eval
    )
    return res


def get_fname(fname):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)


if __name__ == "__main__":
    # Save test data
    res = solve_H_cycle()
    pickle.dump(res, open(FILE_NAME, "wb"))
