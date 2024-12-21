from pydantic.dataclasses import dataclass as _dataclass
from pydantic import ConfigDict
from functools import partial
import numpy as np
from typing import TYPE_CHECKING
import torch
from tqdm.auto import tqdm as _tqdm

Seed = int | np.random.Generator

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    dataclass = partial(_dataclass, config=ConfigDict(arbitrary_types_allowed=True))

tqdm = partial(_tqdm, dynamic_ncols=True, leave=False, ncols=88)


def new_rng(seed: Seed = 0):
    rng = np.random.default_rng(seed)
    return rng.spawn(1)[0]


def new_torch_rng(seed: Seed = 0):
    return torch.Generator().manual_seed(int(new_rng(seed).integers(0, 2**32)))


def get_device(module_or_tensor: torch.nn.Module | torch.Tensor | None = None):
    if module_or_tensor is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(module_or_tensor, torch.Tensor):
        return module_or_tensor.device
    else:
        return next(module_or_tensor.parameters()).device
