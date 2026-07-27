from collections.abc import Callable
from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class CellImmitance:
    Zs_harm_fn: Union[Callable, np.ndarray] # (m, omega) -> complex  OR  (M, Nf) array
    Yg_harm_fn: Union[Callable, np.ndarray] # (m, omega) -> complex  OR  (M, Nf) array
