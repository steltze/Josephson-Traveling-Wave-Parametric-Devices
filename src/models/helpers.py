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


#     # 3 JJs in series
# Series(JJ(L), JJ(L), JJ(L))          # == JJ(3L)

# # 3 JJs in parallel
# Parallel(Series(JJ(L), JJ(L), JJ(L)), Capacitor(C_g))        # == JJ(L/3)

# # Two-mode Δ shunt
# Parallel(Capacitor(C_g), Capacitor(2*C_i))

# # SNAIL (future — just add a SNAIL class with phi^3 potential)
# Series(SNAIL(L_small, L_large, n=3, flux=0.5))
