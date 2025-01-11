from functools import partial, wraps
from typing import TYPE_CHECKING, Mapping

import numpy as np
import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as _dataclass
from tqdm.auto import tqdm as _tqdm

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    dataclass = partial(_dataclass, config=ConfigDict(arbitrary_types_allowed=True))

from dataclasses import fields

Seed = int | np.random.Generator
tqdm = partial(_tqdm, dynamic_ncols=True, leave=False, ncols=88)

tqdm_print = partial(_tqdm.write, end='')


def new_rng(seed: Seed = 0):
    rng = np.random.default_rng(seed)
    return rng.spawn(1)[0]


def new_torch_rng(seed: Seed = 0):
    return torch.Generator().manual_seed(int(new_rng(seed).integers(0, 2**32)))


class DataclassMappingMixin(Mapping):
    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value):
        try:
            setattr(self, key, value)
        except AttributeError:
            raise KeyError(key)

    def __iter__(self):
        for field in fields(self):
            yield field.name

    def __len__(self) -> int:
        return len(fields(self))


def zip_tqdm(*zip_args, **tqdm_kwargs):
    return tqdm(zip(*zip_args), total=len(zip_args[0]), **tqdm_kwargs)


def lmap(fn, iterable):
    return list(map(fn, iterable))
