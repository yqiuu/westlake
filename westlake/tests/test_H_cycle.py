import os
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import torch
import westlake

from utils import get_abs_fname


torch.set_default_dtype(torch.float64)
FILE_NAME = "H_cycle.pickle"
ATOL = 1e-20


def test_H_cycle():
    res_new = solve_H_cycle()
    res_fid = pickle.load(open(get_abs_fname(FILE_NAME), "rb"))
    assert_allclose(res_fid.y, res_new.y, atol=ATOL)


def solve_H_cycle():
    fname = get_fname("H_cycle.h5")
    df_reac = pd.read_hdf(fname, key="reactions")
    df_surf = pd.read_hdf(fname, key="surface_parameters")
    df_spec = pd.read_hdf(fname, key="specie")
    df_act = None

    reaction_matrix = westlake.ReactionMatrix(
        df_reac["reactant_1"], df_reac["reactant_2"], df_reac["products"], df_spec)
    meta_params = westlake.MetaParameters(atol=ATOL)
    westlake.prepare_surface_reaction_params(
        df_reac, df_surf, df_act, reaction_matrix.df_spec, meta_params, specials_barr={'JH': 230.})
    medium = westlake.StaticMedium({'Av': 10., "den_gas": 1e4, "T_gas": 10., "T_dust": 10.})
    reaction_term = westlake.create_two_phase_model(reaction_matrix, df_reac, medium, meta_params)

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
