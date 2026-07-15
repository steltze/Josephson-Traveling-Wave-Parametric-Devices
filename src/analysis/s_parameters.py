from __future__ import annotations

from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_s_parameters(
    S_matrix: np.ndarray,
    freqs: np.ndarray,
    params: Sequence[Tuple[int, int]],
    ax: plt.Axes | None = None,
    freq_scale: float = 1,
    freq_label: str = "Frequency (GHz)",
    ylim: tuple | None = None,
    plot_phase: bool = False,
) -> plt.Axes:
    """
    Plot selected S-parameters in dB vs frequency.

    Parameters
    ----------
    S          : ndarray, shape (Nf, N, N) — complex S-matrix
    freqs      : ndarray, shape (Nf,)      — frequencies in GHz
    params     : list of (i, j) zero-based index pairs to plot
    ax         : existing Axes to draw on; a new figure is created if None
    freq_scale : divisor for the x-axis (default 1 → GHz as-is)
    freq_label : x-axis label string
    ylim       : optional (ymin, ymax) for the y-axis
    plot_phase : if True (default), also plot phase; if False, only magnitude

    Returns
    -------
    ax_mag, ax_phase : the Axes used for plotting (ax_phase is None if plot_phase is False)
    """
    S_matrix = np.asarray(S_matrix)

    if S_matrix.ndim != 3 or S_matrix.shape[1] != S_matrix.shape[2]:
        raise ValueError(f"S_matrix must be (Nf, N, N), got {S_matrix.shape}")

    Nf, N, _ = S_matrix.shape

    freqs = np.asarray(freqs)

    if freqs.shape[0] != Nf:
        raise ValueError(f"freqs length {freqs.shape[0]} does not match Nf={Nf}")

    if ax is None:
        if plot_phase:
            fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        else:
            fig, ax_mag = plt.subplots(1, 1, figsize=(8, 5))
            ax_phase = None
    else:
        ax_mag = ax
        # If user provides only one axis, create a twin axis for phase
        ax_phase = ax.twinx() if plot_phase else None

    x = freqs / freq_scale

    for i, j in params:
        s_param = S_matrix[:, i - 1, j - 1]

        # Magnitude in dB
        db = 20.0 * np.log10(np.abs(s_param))

        ax_mag.plot(x, db, label=f"S{i}{j} | Mag")

        if plot_phase:
            # Phase (unwrapped)
            angles = np.unwrap(np.angle(s_param), axis=0)
            ax_phase.plot(x, angles, linestyle="--", label=f"S{i}{j} | Phase")

    ax_mag.set_ylabel("Magnitude (dB)")

    if plot_phase:
        ax_phase.set_xlabel(freq_label)
        ax_phase.set_ylabel("Phase (rad)")
    else:
        ax_mag.set_xlabel(freq_label)

    if ylim is not None:
        ax_mag.set_ylim(ylim)

    ax_mag.grid(True, alpha=0.3)

    # Combine legends from both axes
    lines1, labels1 = ax_mag.get_legend_handles_labels()
    if plot_phase:
        lines2, labels2 = ax_phase.get_legend_handles_labels()
        ax_mag.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax_mag.legend(lines1, labels1, loc="best")

    return ax_mag, ax_phase
