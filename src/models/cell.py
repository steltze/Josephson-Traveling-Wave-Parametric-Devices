from collections.abc import Callable
from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class CellImmitance:
    theta: float
    Zs0_fn: Callable                        # omega -> complex
    Yg0_fn: Callable                        # omega -> complex
    Zs_harm_fn: Union[Callable, np.ndarray] # (m, omega) -> complex  OR  (M, Nf) array
    Yg_harm_fn: Union[Callable, np.ndarray] # (m, omega) -> complex  OR  (M, Nf) array
