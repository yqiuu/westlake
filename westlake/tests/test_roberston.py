
"""Test the whole framework using the Roberston problem"""
import pickle
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import westlake

from utils import get_dirname


ATOL = 1e-8


def test_roberston_problem():
    res_new = solve_robertson_problem()
    res_fid = pickle.load(open(get_save_name(), "rb"))
    assert_allclose(res_fid.y, res_new.y, atol=ATOL)


def solve_robertson_problem():
    # Define the equation
    reactions = [
        ["A", "", "B", 0.04],
        ["B", "B", "C;B", 3e7],
        ["B", "C", "A;C", 1e4]
    ]
    df_reac = pd.DataFrame(
        reactions, columns=["reactant_1", "reactant_2", "products", "rate"]
    )
    df_spec = westlake.prepare_specie_table(df_reac)
    reaction_term = westlake.create_constant_rate_model(df_reac, df_spec)

    # Solve the problem
    t_begin = 0.
    t_end = 100.
    n_eval = 100
    t_span = (t_begin, t_end)
    t_eval = np.linspace(t_begin, t_end, n_eval)
    ab_0 = np.array([1., 0., 0.])
    res = westlake.solve_ivp_scipy(
        reaction_term, t_span, ab_0, t_eval=t_eval,
        rtol=1e-3, atol=ATOL
    )
    return res


def get_save_name():
    return get_dirname()/Path("data")/Path("data_robertson.pickle")


if __name__ == "__main__":
    # Save test data
    res = solve_robertson_problem()
    pickle.dump(res, open(get_save_name(), "wb"))
