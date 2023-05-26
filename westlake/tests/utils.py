import os


def get_abs_fname(fname):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)