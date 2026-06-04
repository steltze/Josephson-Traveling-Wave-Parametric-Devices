from __future__ import annotations

import numpy as np

from models.cell import CellImmitance


class JTLDiscrete:
    """
    Factory for Josephson Transmission Line unit cells.

    Right-handed (default): series inductor + shunt capacitor.
    Left-handed:            series capacitor + shunt inductor.

    In both topologies:
      - symmetric pi-section: end cells carry half the shunt element
      - first cell has no series element (open left port of the pi)
      - pump modulation acts on the Josephson (reactive) element:
          RH → Zs_harm  (series L modulated)
          LH → Yg_harm  (shunt L modulated)

    Usage
    -----
    sim = Simulation(JTL, cfg)                # right-handed
    sim = Simulation(JTL.left_handed(), cfg)  # left-handed
    """

    handedness: str = "right"

    @classmethod
    def left_handed(cls) -> type:
        """Return a JTL subclass configured as a left-handed transmission line."""

        class LHJTL(cls):
            handedness = "left"

        LHJTL.__name__ = "LHJTL"
        return LHJTL

    @classmethod
    def build(cls, config) -> list[CellImmitance]:
        """
        Build one CellImmitance per unit cell.

        Parameters
        ----------
        config : SimulationConfig

        Returns
        -------
        list[CellImmitance], length config.ncell
        """
        ncell = config.ncell
        ns = np.arange(ncell)
        a = config.cell_size

        ZR = config.Z0 * np.ones(ncell)
        # element values — same parametrisation for both topologies:
        #   RH: L (series),  C (shunt)
        #   LH: C (series),  L (shunt)   ← same numbers, swapped roles
        """
        Add docs on values
        """
        L = ZR / config.omega_cutoff * 2
        C = 2 * 1.0 / (config.omega_cutoff * ZR)
        Cs_jj = 1.0 / (
            config.omega_j**2 * L
        )  # junction self-capacitance (RH) / series cap correction (LH)

        if config.disorder:
            rng = np.random.default_rng(config.disorder_seed)
            lo = 1 - config.disorder_span / 2
            hi = 1 + config.disorder_span / 2
            L *= rng.uniform(lo, hi, ncell)
            C *= rng.uniform(lo, hi, ncell)
            Cs_jj *= rng.uniform(lo, hi, ncell)

        if config.nramp > 0:
            alpha = 4.0 / config.nramp
            ramp_up = 0.5 * (1 + np.tanh(alpha * (ns - config.nramp / 2)))
            ramp_down = 0.5 * (
                1 + np.tanh(alpha * ((ncell - 1 - config.nramp / 2) - ns))
            )
            profile = ramp_up * ramp_down
        else:
            profile = np.ones(ncell)

        epsilons = profile * config.epsilon
        wj = 1.0 / np.sqrt(L * Cs_jj)

        w_p = config.omega_pump
        v_p = config.v_pump
        thetas = w_p / v_p * ns * a

        lh = cls.handedness == "left"
        cells = []
        for i in range(ncell):
            first = i == 0

            if not lh:
                C_end = C[i] / (2.0 if (i == 0 or i == ncell - 1) else 1.0)
                _L, _wj, _eps, _th = L[i], wj[i], epsilons[i], thetas[i]
                _C = C_end
                cells.append(
                    CellImmitance(
                        theta=_th,
                        Zs0_fn=lambda w, L=_L, wji=_wj, f=first: (
                            0.0 if f else 1j * w * L / (1 - w**2 / wji**2)
                        ),
                        Yg0_fn=lambda w, C=_C: 1j * w * C,
                        Zs_harm_fn=lambda m, w, L=_L, wji=_wj, eps=_eps, f=first: (
                            1j * w * L * (eps**m) / ((1 - w**2 / wji**2) ** (m + 1))
                            if (not f)
                            else 0.0
                        ),
                        Yg_harm_fn=lambda m, w: 0.0,
                    )
                )
            else:
                L_end = L[i] / (2.0 if (i == 0 or i == ncell - 1) else 1.0)
                _C, _wj, _eps, _th = C[i], wj[i], epsilons[i], thetas[i]
                _L = L_end
                cells.append(
                    CellImmitance(
                        theta=_th,
                        Zs0_fn=lambda w, C=_C, f=first: 0.0 if f else 1 / (1j * w * C),
                        Yg0_fn=lambda w, L=_L, wji=_wj: (
                            (1 - w**2 / wji**2) / (1j * w * L)
                        ),
                        Zs_harm_fn=lambda m, w: 0j,
                        Yg_harm_fn=lambda m, w, L=_L, eps=_eps: (
                            1j * eps / (2 * w * L) if m == 1 else 0.0
                        ),
                    )
                )
        return cells
