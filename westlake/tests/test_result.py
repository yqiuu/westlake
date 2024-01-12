import westlake

from utils import get_dirname
from numpy.testing import assert_allclose


def test_to_dict():
    res = westlake.load_result(get_dirname()/"data/two_phase.pickle")
    res_new = westlake.Result(**res.to_dict())

    for name in ["message", "success", "nfev", "njev", "nlu", "species", "stages"]:
        assert(getattr(res, name) == getattr(res_new, name))
    for name in ["time", "ab", "den_gas", "coeffs"]:
        assert_allclose(getattr(res, name), getattr(res_new, name), rtol=1e-6, atol=0.)


def test_repr():
    print()
    print(westlake.load_result(get_dirname()/"data/two_phase.pickle"))