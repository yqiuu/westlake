import os
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import torch
import westlake

from utils import get_abs_fname


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
FILE_NAME = "H_cycle.pickle"
RTOL = 1e-5
ATOL = 1e-20


def test_H_cycle():
    res_new = solve_H_cycle()
    res_fid = pickle.load(open(get_abs_fname(FILE_NAME), "rb"))
    assert_allclose(res_fid.y, res_new.y, rtol=RTOL, atol=ATOL)


def solve_H_cycle():
    fname = get_fname("H_cycle.h5")
    df_reac = pd.read_hdf(fname, key="reactions")
    df_spec = westlake.prepare_specie_table(df_reac, pd.read_hdf(fname, key="specie"))
    df_surf = pd.read_hdf(fname, key="surface_parameters")
    df_barr = pd.DataFrame([230.], index=["JH"], columns=["E_barr"])

    ab_0_dict = {'H2': .5,}
    t_start = 0.
    t_end = 1e5
    config = westlake.Config(
        model="two phase", solver="BDF", atol=ATOL,
        t_start=t_start, t_end=t_end,
        Av=10., den_gas=1e4, T_gas=10., T_dust=10.
    )
    reaction_term = westlake.create_astrochem_model(
        df_reac, df_spec, df_surf, config, df_barr=df_barr
    )

    t_begin = 0.
    t_end = 1e5
    n_eval = 100
    t_eval = np.linspace(t_begin, t_end, n_eval)
    res = westlake.solve_rate_equation_astrochem(
        reaction_term, ab_0_dict, df_spec, config, t_eval=t_eval
    )
    return res


def get_fname(fname):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)


if __name__ == "__main__":
    # Save test data
    res = solve_H_cycle()
    pickle.dump(res, open(FILE_NAME, "wb"))
