"""
Physical consistency checks for the TWPA/TWPC S-matrix.

Three independent checks can be run on any (Nf, N, N) S-matrix:

1. Manley-Rowe photon conservation
   For a lossless undepleted-pump system the S-matrix satisfies
       Σᵢ |S[i,j]|² / ωᵢ = 1 / ωⱼ     for every input port j.
   Residual → 0 means photon bookkeeping is exact.

2. Unitarity  (valid for ε = 0, passive lossless network)
   S†S = I.  Any deviation reveals numerical loss or a model error.

3. Reciprocity  (ε = 0 → reciprocal; ε > 0 → non-reciprocal by design)
   ||S − Sᵀ||_F / ||S||_F.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


# ── helpers ──────────────────────────────────────────────────────────────────

def port_frequencies(
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
) -> np.ndarray:
    """
    Angular frequency assigned to each S-matrix port.

    Port ordering mirrors the Floquet state vector:
        [mode_k0_L, mode_k1_L, …, mode_k0_R, mode_k1_R, …]

    Returns
    -------
    port_omegas : ndarray, shape (Nf, N)
    """
    n_modes = len(ks_state)
    N = 2 * n_modes
    port_omegas = np.empty((len(omegas), N))
    for i, k in enumerate(ks_state):
        freq = omegas + k * omega_pump
        port_omegas[:, i] = freq            # left port
        port_omegas[:, n_modes + i] = freq  # right port
    return port_omegas


# ── check functions ───────────────────────────────────────────────────────────

def manley_rowe_residual(
    S: np.ndarray,
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
) -> np.ndarray:
    """
    Fractional Manley-Rowe residual per frequency per input port.

    Photon conservation requires
        Σᵢ |S[i,j]|² / ωᵢ  =  1 / ωⱼ

    so the returned quantity  (lhs − rhs) / rhs  should be ≈ 0.

    Parameters
    ----------
    S : (Nf, N, N) — complex S-matrix, S[f, output, input]
    omegas : (Nf,) — signal angular frequencies [rad/s]
    omega_pump : pump angular frequency [rad/s]
    ks_state : sideband indices, e.g. [0, 1]

    Returns
    -------
    residual : (Nf, N) — fractional error; 0 = exact conservation
    """
    w = port_frequencies(omegas, omega_pump, ks_state)  # (Nf, N)
    Sabsq = np.abs(S) ** 2                              # (Nf, N, N)

    # Sum |S[i,j]|² / ω_i over output ports i (axis -2 = rows)
    weighted_col_sum = (Sabsq / w[:, :, None]).sum(axis=-2)   # (Nf, N)
    reference = 1.0 / w                                        # (Nf, N)
    return (weighted_col_sum - reference) / reference


def unitarity_residual(S: np.ndarray) -> np.ndarray:
    """
    Frobenius deviation of S†S from the identity, normalised by N.

    Returns
    -------
    residual : (Nf,) — should be ≈ 0 for a passive lossless network (ε = 0)
    """
    Nf, N, _ = S.shape
    SdagS = S.conj().swapaxes(-1, -2) @ S
    eye = np.broadcast_to(np.eye(N, dtype=complex), (Nf, N, N))
    return np.linalg.norm(SdagS - eye, axis=(-1, -2)) / N


def reciprocity_residual(S: np.ndarray) -> np.ndarray:
    """
    Normalised non-reciprocity ||S − Sᵀ||_F / ||S||_F.

    Returns
    -------
    residual : (Nf,) — ≈ 0 for ε = 0; grows with ε (directional pump)
    """
    diff = np.linalg.norm(S - S.swapaxes(-1, -2), axis=(-1, -2))
    norm = np.linalg.norm(S, axis=(-1, -2))
    return diff / np.maximum(norm, 1e-300)


def power_sum(S: np.ndarray) -> np.ndarray:
    """
    Total output power Σᵢ |S[i,j]|² for each input port j.

    For ε = 0 (passive, lossless): = 1.
    For ε > 0 (active): ≥ 1 at frequencies where the pump transfers energy.

    Returns
    -------
    psum : (Nf, N)
    """
    return (np.abs(S) ** 2).sum(axis=-2)


# ── combined plot ─────────────────────────────────────────────────────────────

def plot_verification(
    S: np.ndarray,
    freqs: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
    epsilon: float | None = None,
) -> plt.Figure:
    """
    Four-panel verification figure.

    Panel 1 — Manley-Rowe residual (log scale) per input port.
    Panel 2 — Total output power per input port; dashed line at 1 (passive limit).
    Panel 3 — Unitarity deviation ||S†S − I||_F / N (log scale).
    Panel 4 — Non-reciprocity ||S − Sᵀ||_F / ||S||_F (log scale).

    Parameters
    ----------
    S     : (Nf, N, N) complex
    freqs : (Nf,) in Hz
    omega_pump : pump angular frequency [rad/s]
    ks_state   : sideband indices, e.g. [0, 1]
    epsilon    : modulation depth (annotation only)
    """
    omegas = freqs * 2 * np.pi
    f_GHz = freqs / 1e9
    N = S.shape[1]
    n_modes = len(ks_state)

    mr   = manley_rowe_residual(S, omegas, omega_pump, ks_state)  # (Nf, N)
    ps   = power_sum(S)                                            # (Nf, N)
    u    = unitarity_residual(S)                                   # (Nf,)
    rec  = reciprocity_residual(S)                                 # (Nf,)

    port_labels = (
        [f"k={k} L" for k in ks_state] +
        [f"k={k} R" for k in ks_state]
    )

    eps_str = f"ε = {epsilon:.3g}" if epsilon is not None else ""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"S-matrix verification  {eps_str}", fontsize=13)

    # ── panel 1: Manley-Rowe ─────────────────────────────────────────────────
    ax = axes[0, 0]
    for j in range(N):
        ax.semilogy(f_GHz, np.maximum(np.abs(mr[:, j]), 1e-16),
                    label=f"port {j+1} ({port_labels[j]})")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|fractional residual|")
    ax.set_title("Manley-Rowe  Σᵢ|Sᵢⱼ|²/ωᵢ = 1/ωⱼ")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── panel 2: power sum ───────────────────────────────────────────────────
    ax = axes[0, 1]
    for j in range(N):
        ax.plot(f_GHz, ps[:, j], label=f"port {j+1} ({port_labels[j]})")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="passive = 1")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Σᵢ |Sᵢⱼ|²")
    ax.set_title("Total output power  (>1 where pump adds energy)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── panel 3: unitarity ───────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.semilogy(f_GHz, np.maximum(u, 1e-16))
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("||S†S − I||_F / N")
    ax.set_title("Unitarity deviation  (→ 0 for ε = 0)")
    ax.grid(True, alpha=0.25)

    # ── panel 4: reciprocity ─────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.semilogy(f_GHz, np.maximum(rec, 1e-16))
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("||S − Sᵀ||_F / ||S||_F")
    ax.set_title("Non-reciprocity  (→ 0 for ε = 0)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig
