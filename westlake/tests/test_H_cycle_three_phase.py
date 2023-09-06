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
ATOL = 1e-20


def test_H_cycle():
    res_new = solve_H_cycle()
    res_fid = pickle.load(open(get_abs_fname(FILE_NAME), "rb"))
    assert_allclose(res_fid.y, res_new.y, atol=ATOL)


def solve_H_cycle():
    fname = get_fname("H_cycle_three_phase.h5")
    df_reac = pd.read_hdf(fname, key="reactions")
    df_surf = pd.read_hdf(fname, key="surface_parameters")
    df_spec = pd.read_hdf(fname, key="species")
    df_barr = pd.DataFrame([230.], index=["JH"], columns=["E_barr"])

    meta_params = westlake.MetaParameters(atol=ATOL, ab_0_min=1e-40)
    medium = westlake.StaticMedium({'Av': 10., "den_gas": 1e4, "T_gas": 17., "T_dust": 17.})
    reaction_term = westlake.create_three_phase_model(
        df_reac, df_spec, df_surf, medium, meta_params, df_barr=df_barr
    )

    ab_0 = westlake.dervie_initial_abundances({'H2': .5,}, df_spec, meta_params)
    t_begin = 0.
    t_end = 1e5
    n_eval = 100
    t_span = (t_begin, t_end)
    t_eval = np.linspace(t_begin, t_end, n_eval)
    res = westlake.solve_kinetic(reaction_term, t_span, ab_0, meta_params, t_eval=t_eval)
    return res


def get_fname(fname):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)


if __name__ == "__main__":
    # Save test data
    res = solve_H_cycle()
    pickle.dump(res, open(FILE_NAME, "wb"))
