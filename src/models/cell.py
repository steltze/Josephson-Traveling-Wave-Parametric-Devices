from collections.abc import Callable
from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class CellImmitance:
    Zs_harm_fn: Union[Callable, np.ndarray]  # (m, omega) -> complex  OR  (M, Nf) array
    Yg_harm_fn: Union[Callable, np.ndarray]  # (m, omega) -> complex  OR  (M, Nf) array
    Zs_slot_fn: Union[Callable, np.ndarray] = None
    Yg_slot_fn: Union[Callable, np.ndarray] = None
    Yi_slot_coupling_fn: Union[Callable, np.ndarray] = None
