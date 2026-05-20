from dataclasses import dataclass
import numpy as np


@dataclass
class Immittance:
    """

        Parameters
    ----------
    Z0 : float
        Port impedance in Ohms. Default 50 Ω.
    """

    Z0: float = 50.0
