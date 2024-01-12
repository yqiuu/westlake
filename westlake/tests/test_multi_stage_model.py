from pathlib import Path

from numpy.testing import assert_allclose
import pandas as pd
import westlake

from utils import get_dirname


RTOL = 1e-5
ATOL = 1e-20


def test_model():
    res = solve_model()
    res_fid = westlake.load_result(get_save_name())
    assert res.stages == res_fid.stages
    assert_allclose(res.time, res_fid.time, rtol=RTOL)
    for spec in res_fid.species:
        assert_allclose(res[spec][-1], res_fid[spec][-1], rtol=RTOL, atol=ATOL)


def solve_model():
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
        model="two phase",
        save_rate_coeffs=True,
        dtg_mass_ratio=westlake.fixed_dtg_mass_ratio(ab_0_dict['He']),
        H2_shielding="Lee+1996",
        CO_shielding="Lee+1996",
        use_scipy_solver=False,
        rtol=1e-4,
        atol=1e-25,
    )

    medium_list = [
        westlake.Medium(config, Av=2.7, den_gas=1.3e4, T_gas=11.2, T_dust=15.3),
        westlake.Medium(config, Av=5.2, den_gas=2.1e4, T_gas=15.2, T_dust=17.2)
    ]
    t_span = (1e-10, 1e3, 1.7e3)
    reaction_term = westlake.create_astrochem_model(df_reac, df_spec, df_surf, config)
    res = westlake.solve_rate_equation_astrochem(
        reaction_term, ab_0_dict, df_spec, config,
        medium_list=medium_list, t_span=t_span
    )
    return res


def get_save_name():
    return get_dirname()/Path("data")/Path("multi_stage_model.pickle")


if __name__ == "__main__":
    res = solve_model()
    westlake.save_result(res, get_save_name())
