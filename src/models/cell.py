from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class CellImmitance:
    theta: float
    Zs0_fn: Callable  # omega -> complex
    Yg0_fn: Callable  # omega -> complex
    Zs_harm_fn: Callable  # (m: int, omega) -> complex  (m-th Fourier coeff of Zs)
    Yg_harm_fn: Callable  # (m: int, omega) -> complex
