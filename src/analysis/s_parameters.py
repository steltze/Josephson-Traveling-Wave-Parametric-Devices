from __future__ import annotations

from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_s_parameters(
    S_matrix: np.ndarray,
    freqs: np.ndarray,
    params: Sequence[Tuple[int, int]],
    ax: plt.Axes | None = None,
    freq_scale: float = 1e9,
    freq_label: str = "Frequency (GHz)",
    ylim: tuple | None = None,
) -> plt.Axes:
    """
    Plot selected S-parameters in dB vs frequency.

    Parameters
    ----------
    S          : ndarray, shape (Nf, N, N) — complex S-matrix
    freqs      : ndarray, shape (Nf,)      — frequencies in Hz
    params     : list of (i, j) zero-based index pairs to plot
    ax         : existing Axes to draw on; a new figure is created if None
    freq_scale : divisor for the x-axis (default 1e9 → GHz)
    freq_label : x-axis label string
    ylim       : optional (ymin, ymax) for the y-axis

    Returns
    -------
    ax : the Axes used for plotting
    """
    S_matrix = np.asarray(S_matrix)

    if S_matrix.ndim != 3 or S_matrix.shape[1] != S_matrix.shape[2]:
        raise ValueError(f"S_matrix must be (Nf, N, N), got {S_matrix.shape}")

    Nf, N, _ = S_matrix.shape

    freqs = np.asarray(freqs)

    if freqs.shape[0] != Nf:
        raise ValueError(f"freqs length {freqs.shape[0]} does not match Nf={Nf}")

    if ax is None:
        _, ax = plt.subplots()

    x = freqs / freq_scale
    for i, j in params:
        db = 20.0 * np.log10(np.abs(S_matrix[:, i, j]))
        ax.plot(x, db, label=f"S_matrix{i + 1}{j + 1}")

    ax.set_xlabel(freq_label)
    ax.set_ylabel("Magnitude (dB)")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax
