
"""Test the whole framework using the Roberston problem"""

import os
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import torch
import westlake

from utils import get_abs_fname


torch.set_default_dtype(torch.float64)
FILE_NAME = "data_robertson.pickle"
ATOL = 1e-8


def test_roberston_problem():
    res_new = solve_robertson_problem()
    res_fid = pickle.load(open(get_abs_fname(FILE_NAME), "rb"))
    assert_allclose(res_fid.y, res_new.y, atol=ATOL)


def solve_robertson_problem():
    # Define the equation
    reactions = [
        ["A", "", "B", 0.04],
        ["B", "B", "C;B", 3e7],
        ["B", "C", "A;C", 1e4]
    ]
    df_reac = pd.DataFrame(reactions, columns=["reactant_1", "reactant_2", "products", "rate"])
    _, rmat_1st, rmat_2nd = westlake.create_reaction_matrices(df_reac["reactant_1"], df_reac["reactant_2"], df_reac["products"])
    rate_1st = westlake.ConstantReactionRate(rmat_1st, df_reac["rate"].values)
    rate_2nd = westlake.ConstantReactionRate(rmat_2nd, df_reac["rate"].values)
    reaction_term = westlake.ReactionTerm(rmat_1st, rate_1st, rmat_2nd, rate_2nd)

    # Solve the problem
    t_begin = 0.
    t_end = 100.
    n_eval = 100
    t_span = (t_begin, t_end)
    t_eval = np.linspace(t_begin, t_end, n_eval)
    ab_0 = np.array([1., 0., 0.])
    res = westlake.solve_kinetic(reaction_term, t_span, ab_0, t_eval=t_eval, atol=ATOL)
    return res


if __name__ == "__main__":
    # Save test data
    res = solve_robertson_problem()
    pickle.dump(res, open(FILE_NAME, "wb"))
