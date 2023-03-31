from dataclasses import dataclass

import numpy as np
import torch


class KeyTensor:
    """A tensor whose last dimension can be accessed by a list of keys."""
    def __init__(self, keys, data):
        self._lookup = {key: idx for idx, key in enumerate(keys)}
        self._data = data

    def get(self, keys=None):
        if keys is None:
            return self._data

        if isinstance(keys, str):
            return self._data[..., self._lookup[keys]]

        inds = [self._lookup[key] for key in keys]
        return self._data[..., inds]

    def register_buffer(self, module, name):
        name_ = f"{name}_"
        module.register_buffer(name_, self._data)
        setattr(module, name, KeyTensor(self._lookup, getattr(module, name_)))


def data_frame_to_key_tensor(df, **kwargs):
    return KeyTensor(df.columns.values.astype(str), torch.tensor(df.values, **kwargs))
