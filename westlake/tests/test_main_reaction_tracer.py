import pandas as pd
import westlake
import pytest

from utils import get_dirname


@pytest.mark.parametrize("is_instant", (True, False))
def test_tracer(is_instant):
    fname = get_save_name(is_instant)
    df_prod_fid = pd.read_hdf(fname, "prod")
    df_dest_fid = pd.read_hdf(fname, "dest")

    df_prod, df_dest = run_tracer(is_instant)

    assert df_prod_fid.equals(df_prod)
    assert df_dest_fid.equals(df_dest)


def run_tracer(is_instant):
    dirname = get_dirname()
    df_reac = pd.read_csv(dirname/"inputs"/"reactions.csv", na_filter=False)
    df_spec = pd.read_csv(dirname/"inputs"/"species.csv", index_col="specie")
    res = westlake.load_result(dirname/"data"/"two_phase.pickle")
    tracer = westlake.MainReactionTracer.from_result(res, df_reac, df_spec)

    specie = "CO"
    t_start = 0. # [yr]
    t_end = 1. # [yr]
    if is_instant:
        df_prod, df_dest = tracer.trace_instant(specie, t_end, percent_cut=1., rate_cut=0.)
    else:
        df_prod, df_dest = tracer.trace_period(specie, t_start, t_end, percent_cut=1., rate_cut=0.)
    return df_prod, df_dest


def get_save_name(is_instant):
    dirname = get_dirname()/"data"
    if is_instant:
        fname = dirname/"trace_instant.hdf5"
    else:
        fname = dirname/"trace_period.hdf5"
    return fname


if __name__ == "__main__":
    dirname = get_dirname()
    for is_instant in (True, False):
        df_prod, df_dest = run_tracer(is_instant)
        with pd.HDFStore(get_save_name(is_instant), "w") as fp:
            df_prod.to_hdf(fp, "prod")
            df_dest.to_hdf(fp, "dest")