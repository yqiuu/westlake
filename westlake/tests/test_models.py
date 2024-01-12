from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import westlake
import pytest

from utils import get_dirname


RTOL = 1e-5
ATOL = 1e-20


@pytest.mark.parametrize("model", ("two phase", "three phase"))
def test_model(model):
    res = solve_model(model)
    res_fid = westlake.load_result(get_save_name(model))
    assert_allclose(res.time, res_fid.time, rtol=RTOL)
    for spec in res_fid.species:
        assert_allclose(res[spec], res_fid[spec], rtol=RTOL, atol=ATOL)


def solve_model(model):
    dirname = get_dirname()/Path("inputs")
    df_reac = pd.read_csv(dirname/Path("reactions.csv"), na_filter=False)
    df_spec = pd.read_csv(dirname/Path("species.csv"), index_col="specie")
    df_surf = pd.read_csv(dirname/Path("surface_parameters.csv"), index_col="specie")

    # Initial condition
    ab_0_dict = {
        'H': 0.0,
        'H2': 0.5,
        'He': 0.09,
        'O': 2.4e-4,
        'C+': 1.7e-04,
        'S+': 8e-08,
    }
    # Create config
    config = westlake.Config(
        model=model,
        save_rate_coeffs=True,
        dtg_mass_ratio=westlake.fixed_dtg_mass_ratio(ab_0_dict['He']),
        H2_shielding="Lee+1996",
        CO_shielding="Lee+1996",
        Av=2.7,
        den_gas=1.3e4,
        T_gas=11.2,
        T_dust=15.3,
        uv_flux=1.9,
        t_start=1e-10,
        t_end=1e4,
        use_scipy_solver=False,
        rtol=1e-4,
        atol=1e-25,
    )

    n_eval = 100
    t_eval = np.logspace(np.log10(config.t_start), np.log10(config.t_end), n_eval)
    reaction_term = westlake.create_astrochem_model(df_reac, df_spec, df_surf, config)
    res = westlake.solve_rate_equation_astrochem(
        reaction_term, ab_0_dict, df_spec, config, t_eval=t_eval
    )
    return res


def get_save_name(model):
    return get_dirname()/Path("data")/Path("{}.pickle".format(model.replace(" ", "_")))


if __name__ == "__main__":
    # Save test data
    for model in ["two phase", "three phase"]:
        res = solve_model(model)
        westlake.save_result(res, get_save_name(model))
