

from dataclasses import dataclass
import numpy as np

@dataclass
class Immittance:
    """
    Configuration for a Josephson transmission line simulation.

    The simulator accepts physical circuit parameters and derives everything
    else.

        Parameters
    ----------
    Z0 : float
        Port impedance in Ohms. Default 50 Ω.

    ncell : int
        Number of unit cells. More cells → sharper, deeper isolation gap
        but slower simulation.
    """

    L: float = 50.0

    C: float = 500

    in_series: bool = True

    
