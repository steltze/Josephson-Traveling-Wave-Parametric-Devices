"""
Physical consistency checks for the TWPA/TWPC matrices.

1. Photon-flux conservation (diagonal and full pseudo-unitarity)

2. Transfer matrix determinant (for a lossless material) should be 1.

3. Numerical-stability diagnostics for the cascade itself: split-cascade
   associativity, Redheffer-star conditioning, an ABCD-product cross-check,
   backend agreement, and a gain-vs-cell-count trend plot. These target
   error that *accumulates* across the Nc-cell cascade -- the kind that a
   per-cell determinant check cannot see, since it only tests one cell at
   a time.
"""

from __future__ import annotations

import itertools

import numpy as np
import matplotlib.pyplot as plt

from backends import Backend
from logger import get_logger

log = get_logger(__name__)


def _get_port_frequencies(
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
        port_omegas[:, i] = freq  # left port
        port_omegas[:, n_modes + i] = freq  # right port
    return port_omegas


def check_photon_flux_conservation(
    S_ph: np.ndarray,
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
) -> np.ndarray:
    """
    eta-weighted power sum for a photon-flux-normalized S-matrix.

    ``S_ph`` must come from photon-flux normalization (e.g.
    ``Simulation.get_s_matrix(normalize=True)`` or
    ``SMatrix.normalize_photon_flux``). A port's signed frequency
    ωₖ = ω + k·ω_pump can be negative (an idler / down-converted sideband,
    common whenever ks_state contains a negative k) — such ports are
    pseudo-unitary partners of the positive-frequency ports, not ordinary
    channels. A *plain* Σᵢ|S_ph[i,j]|² is not conserved for them and can
    look wildly "overestimated" wherever that port carries real parametric
    gain. The conserved quantity (Manley-Rowe photon number, exact for a
    lossless-junction line even with gain) is instead:

        Σᵢ ηᵢ |S_ph[i,j]|²  =  ηⱼ,      ηₖ = sign(ωₖ)

    Returns
    -------
    check : (Nf, N) — should equal ηⱼ (±1) for every input port j; deviation
        indicates real dissipation or a port-labeling/normalization bug.
    """
    w = _get_port_frequencies(omegas, omega_pump, ks_state)  # (Nf, N) signed
    eta = np.sign(w)
    Sabsq = np.abs(S_ph) ** 2  # (Nf, N, N)
    return (Sabsq * eta[:, :, np.newaxis]).sum(axis=1)  # sum over output i


def check_reflection_vs_transmission(
    S_ph: np.ndarray,
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
    input_port: int,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the conserved eta-weighted power for one input port into what
    came back out the left/input side (reflection) vs what exited the
    right/output side (transmission).

    Whenever a transmission curve flattens or a "gap" stops deepening as
    some parameter grows, this answers *where the power went* instead of
    guessing: since `check_photon_flux_conservation` already establishes
    Sigma_i eta_i |S_ph[i,j]|^2 = eta_j for a lossless-junction line
    (dissipation-free even with parametric gain), R + T == eta_j is an
    exact identity -- if reflection is rising toward |eta_j| while
    transmission falls, the missing power is provably being reflected,
    not absorbed or lost to some unmodeled channel.

    Parameters
    ----------
    S_ph : ndarray, shape (Nf, N, N) -- photon-flux-normalized S-matrix.
    input_port : the column (input) port to analyze; port < N//2 is a
        left-side (input) port, port >= N//2 is a right-side (output) port.

    Returns
    -------
    R, T : ndarray, shape (Nf,) -- eta-weighted reflected and transmitted
        power fractions for `input_port`. R + T should equal
        eta[input_port] (+1 or -1) to within `tolerance`.

    Note
    ----
    This bundles same-mode and mode-converted contributions together by
    port *side* alone (e.g. "signal reflected as signal" and "signal
    converted to idler and sent back out the input side" both land in R).
    Rising R does not by itself distinguish plain reflection from
    parametric mode conversion -- use `get_power_flow` for the unbundled
    per-port breakdown when that distinction matters.
    """
    n_modes = len(ks_state)
    w = _get_port_frequencies(omegas, omega_pump, ks_state)  # (Nf, N)
    eta = np.sign(w)
    Sabsq = np.abs(S_ph[:, :, input_port]) ** 2  # (Nf, N)
    weighted = Sabsq * eta  # (Nf, N)
    R = weighted[:, :n_modes].sum(axis=1)  # left-side ports: reflection
    T = weighted[:, n_modes:].sum(axis=1)  # right-side ports: transmission

    residual = np.abs(R + T - eta[:, input_port])
    if (residual >= tolerance).any():
        log.error(
            f"Reflection+transmission mismatch for port {input_port}: "
            f"max |R+T-eta| = {residual.max():.3e} (>= {tolerance})."
        )
    else:
        log.test(
            f"Reflection/transmission split for port {input_port} pass "
            f"(max |R+T-eta| = {residual.max():.3e})."
        )
    return R, T


def plot_reflection_vs_transmission(
    freqs: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """
    Plot reflected/transmitted power fraction vs frequency from
    `check_reflection_vs_transmission`'s output.

    Reflection is drawn solid, transmission dashed, sharing a color per
    call so multiple sweeps (e.g. one per epsilon) overlay cleanly.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    (line,) = ax.plot(freqs, R, "-", label=f"R{f' ({label})' if label else ''}")
    ax.plot(freqs, T, "--", color=line.get_color(), label=f"T{f' ({label})' if label else ''}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Power fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def get_power_flow(
    S_ph: np.ndarray,
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
    input_port: int,
) -> np.ndarray:
    """
    Unbundled eta-weighted power delivered to every output port from
    `input_port` -- one value per port, not pre-summed by side.

    `check_reflection_vs_transmission` bundles same-mode and
    mode-converted contributions together by port side alone, which can
    hide *which* physical process is responsible for a change: "signal
    reflected as signal" and "signal converted to idler and sent back out
    the input side" both land in its R bucket even though one is plain
    reflection and the other is parametric mode conversion. This returns
    the per-port values unbundled so the two can be told apart (e.g. by
    port side *and* by which ks_state sideband each port carries).

    Returns
    -------
    power : ndarray, shape (Nf, N) -- eta_i * |S_ph[i, input_port]|^2 for
        every output port i. Sums (over i) to eta[input_port] exactly
        (the same identity `check_photon_flux_conservation` checks).
    """
    w = _get_port_frequencies(omegas, omega_pump, ks_state)  # (Nf, N)
    eta = np.sign(w)
    return (np.abs(S_ph[:, :, input_port]) ** 2) * eta


def plot_power_flow(
    freqs: np.ndarray,
    power: np.ndarray,
    ks_state: list[int],
    input_port: int,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot the per-port breakdown from `get_power_flow`, one curve per
    output port, labeled by side (reflected/transmitted) and sideband so
    mode conversion is visually distinguishable from plain reflection.
    """
    n_modes = len(ks_state)
    in_k = ks_state[input_port % n_modes]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for i in range(power.shape[1]):
        side = "reflected" if i < n_modes else "transmitted"
        k = ks_state[i % n_modes]
        mode = "same mode" if k == in_k else "converted"
        ax.plot(freqs, power[:, i], label=f"{side}, k={k} ({mode})")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"$\eta_i\,|S_{i,\mathrm{in}}|^2$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def check_transfer_matrix_determinant(
    T_grid: np.ndarray,  # (Nf, Nc, N, N)
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the (Nf, Nc) indices where |det(T) - 1| >= tolerance.
    """
    dets = np.abs(np.linalg.det(T_grid))  # (Nf, Nc)
    violating = np.abs(dets - 1.0) >= tolerance
    nf_idx, nc_idx = np.where(violating)
    if nf_idx.size:
        log.error(f"{nf_idx.size} cells with |det(T)-1| >= {tolerance}.")
    else:
        log.test("Determinant check pass!")
    return nf_idx, nc_idx


def check_pseudo_unitarity(
    S_ph: np.ndarray,
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full Manley-Rowe identity S_ph^H @ diag(eta) @ S_ph == diag(eta).

    ``check_photon_flux_conservation`` only tests the *diagonal* of this
    identity (the eta-weighted column norms). Off-diagonal residuals can
    be nonzero even when every column norm looks fine, since a sum of
    squared magnitudes hides compensating errors that only show up in the
    full matrix product -- this is a strictly stronger test at the same
    cost.

    Parameters
    ----------
    S_ph : ndarray, shape (Nf, N, N) -- photon-flux-normalized S-matrix
        (see ``check_photon_flux_conservation``).

    Returns
    -------
    nf_idx, i_idx, j_idx : indices where
        |(S_ph^H diag(eta) S_ph)_ij - eta_i * delta_ij| >= tolerance.
    """
    w = _get_port_frequencies(omegas, omega_pump, ks_state)  # (Nf, N)
    eta = np.sign(w)
    lhs = np.einsum("fij,fi,fik->fjk", np.conj(S_ph), eta, S_ph)
    target = np.zeros_like(lhs)
    idx = np.arange(S_ph.shape[-1])
    target[:, idx, idx] = eta
    residual = np.abs(lhs - target)
    violating = residual >= tolerance
    nf_idx, i_idx, j_idx = np.where(violating)
    if nf_idx.size:
        log.error(
            f"{nf_idx.size} entries with pseudo-unitarity residual >= {tolerance} "
            f"(max {residual.max():.3e})."
        )
    else:
        log.test(f"Pseudo-unitarity check pass (max residual {residual.max():.3e}).")
    return nf_idx, i_idx, j_idx


def check_cascade_associativity(
    S_cells: np.ndarray,
    backend: Backend | str | None = None,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the Nc cells in half, cascade each half, and Redheffer-star the
    two halves together; compare against cascading all Nc cells in one
    sequential pass.

    The Redheffer star product is associative, so the two results should
    agree to near machine precision regardless of gain. A mismatch that
    grows with Nc pinpoints accumulated rounding error in the sequential
    reduction (`Backend.cascade_all` / `Simulation.get_s_matrix`) rather
    than a physical effect -- and directly shows at what cell count it
    starts to matter.

    Parameters
    ----------
    S_cells : ndarray, shape (Nf, Nc, N, N) -- per-cell S-matrices, ordered
        left (input) to right (output).

    Returns
    -------
    nf_idx, i_idx, j_idx : indices where
        |S_direct - S_split| >= tolerance.
    """
    from numerical_solver.s_matrix import cascade_all, redheffer_star

    Nc = S_cells.shape[1]
    if Nc < 2:
        raise ValueError(f"Need at least 2 cells to test associativity, got Nc={Nc}.")
    mid = Nc // 2

    S_direct = cascade_all(S_cells, backend=backend)
    S_left = cascade_all(S_cells[:, :mid], backend=backend)
    S_right = cascade_all(S_cells[:, mid:], backend=backend)
    S_split = redheffer_star(S_right, S_left, backend=backend)

    residual = np.abs(S_direct - S_split)
    violating = residual >= tolerance
    nf_idx, i_idx, j_idx = np.where(violating)
    if nf_idx.size:
        log.error(
            f"{nf_idx.size} entries with split-cascade mismatch >= {tolerance} "
            f"(max {residual.max():.3e})."
        )
    else:
        log.test(f"Cascade associativity check pass (max mismatch {residual.max():.3e}).")
    return nf_idx, i_idx, j_idx


def check_cascade_conditioning(
    S_cells: np.ndarray,
    backend: Backend | str | None = None,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Condition number of (I - S2_11 @ S1_22) at every Redheffer-star
    reduction step across the cascade.

    This is the linear solve inside each `redheffer_star` call (see
    `backends.numpy_backend.NumpyBackend.redheffer_star`). As internal
    reflections build up with gain and cell count, this matrix can
    approach singular -- a direct mechanism for the S-cascade saturating
    or losing precision well before any per-cell determinant would flag
    it. Track this vs Ncell/epsilon to locate the onset of instability.

    Parameters
    ----------
    S_cells : ndarray, shape (Nf, Nc, N, N) -- per-cell S-matrices, ordered
        left (input) to right (output).
    tolerance : target relative-error budget. Solving a linear system with
        condition number cond(A) loses about log10(cond(A)) decimal digits,
        so the predicted relative error is ~cond(A) * eps_machine; this
        flags cond once that predicted error would exceed `tolerance`.
        (Comparing cond directly against 1/eps_machine, as an earlier
        version of this check did, only fires once *all* ~16 digits of
        double precision are already gone -- a check that late is not an
        early warning, it's a post-mortem.)

    Returns
    -------
    cond : ndarray, shape (Nf, Nc - 1) -- cond(I - S2_11 @ S1_22) at each
        of the Nc-1 reduction steps, in cascade order.
    """
    from numerical_solver.s_matrix import redheffer_star

    Nf, Nc, N, _ = S_cells.shape
    if Nc < 2:
        raise ValueError(f"Need at least 2 cells to track conditioning, got Nc={Nc}.")
    k = N // 2

    cond = np.empty((Nf, Nc - 1))
    total = S_cells[:, 0]
    eye_k = np.eye(k)[None]
    for c in range(1, Nc):
        s2 = S_cells[:, c]
        s1 = total
        A1 = eye_k - s2[:, :k, :k] @ s1[:, k:, k:]
        cond[:, c - 1] = np.linalg.cond(A1)
        total = redheffer_star(s2, s1, backend=backend)

    max_cond = cond.max()
    eps_machine = np.finfo(float).eps
    predicted_error = max_cond * eps_machine
    if predicted_error >= tolerance:
        log.error(
            f"Cascade conditioning reached {max_cond:.3e} "
            f"(predicted relative error {predicted_error:.3e} >= tolerance={tolerance:.3e})."
        )
    else:
        log.test(
            f"Cascade conditioning check pass "
            f"(max cond {max_cond:.3e}, predicted relative error {predicted_error:.3e})."
        )
    return cond


def check_abcd_product_vs_s_cascade(
    T_grid: np.ndarray,
    S_total: np.ndarray,
    Z0: float = 50.0,
    backend: Backend | str | None = None,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Independent cross-check: multiply the per-cell ABCD matrices directly
    (T_total = T_0 @ T_1 @ ... @ T_{Nc-1}), convert that single product to
    S once, and compare against S_total -- the S-matrix obtained by
    converting each cell to S individually and cascading via Redheffer
    star (the actual path used by `Simulation.get_s_matrix`).

    T_total's entries grow ~exp(gain) with Nc/epsilon, so this check is
    itself precision-limited at high gain: use it to cross-validate at
    moderate Nc/epsilon, and use the accompanying spectral norm of
    T_total to tell whether *this* path has already lost precision before
    trusting a mismatch as evidence against the S-cascade.

    Parameters
    ----------
    T_grid : ndarray, shape (Nf, Nc, N, N) -- per-cell ABCD matrices.
    S_total : ndarray, shape (Nf, N, N) -- S-matrix from the Redheffer-star
        cascade (e.g. ``Simulation.get_s_matrix().array``).

    Returns
    -------
    diff : ndarray, shape (Nf, N, N) -- |S_from_abcd_product - S_total|.
    T_norm : ndarray, shape (Nf,) -- spectral norm (largest singular value)
        of T_total, diagnostic for when the raw ABCD product itself has
        lost precision.
    """
    from numerical_solver.s_matrix import ABCD_to_S

    Nc = T_grid.shape[1]
    T_total = T_grid[:, 0]
    for c in range(1, Nc):
        T_total = T_total @ T_grid[:, c]

    S_from_abcd = ABCD_to_S(T_total, Z0, backend=backend)
    diff = np.abs(S_from_abcd - S_total)
    T_norm = np.linalg.svd(T_total, compute_uv=False)[..., 0]

    max_diff = diff.max()
    if max_diff >= tolerance:
        log.error(
            f"ABCD-product vs S-cascade mismatch: max |diff| = {max_diff:.3e} "
            f"(max ||T_total|| = {T_norm.max():.3e})."
        )
    else:
        log.test(f"ABCD-product vs S-cascade check pass (max |diff| = {max_diff:.3e}).")
    return diff, T_norm


def check_backend_agreement(
    S_cells: np.ndarray,
    backends: tuple[str, ...] = ("numpy", "numba"),
    tolerance: float = 1e-9,
) -> dict[str, np.ndarray]:
    """
    Cascade the same per-cell S-matrices through every backend in
    `backends` and compare pairwise.

    Every backend implements identical numerics -- numba only fuses the
    same sequential Redheffer-star reduction into one compiled kernel
    (see `backends.numba_backend`), it doesn't change the math. So any
    backend-to-backend divergence that grows with Nc is a floating-point
    ordering / instability signal, not an implementation bug in either
    backend.

    Parameters
    ----------
    S_cells : ndarray, shape (Nf, Nc, N, N) -- per-cell S-matrices, ordered
        left (input) to right (output).
    backends : names to compare; unavailable backends (e.g. numba not
        installed) are skipped with a logged warning.

    Returns
    -------
    diffs : dict mapping "{a} vs {b}" -> |S_a - S_b|, shape (Nf, N, N), for
        every pair of backends that both imported successfully.
    """
    from numerical_solver.s_matrix import cascade_all

    results: dict[str, np.ndarray] = {}
    for name in backends:
        try:
            results[name] = cascade_all(S_cells, backend=name)
        except ImportError as exc:
            log.error(f"Backend {name!r} unavailable, skipping ({exc}).")

    diffs: dict[str, np.ndarray] = {}
    for a, b in itertools.combinations(results, 2):
        diff = np.abs(results[a] - results[b])
        max_diff = diff.max()
        key = f"{a} vs {b}"
        diffs[key] = diff
        if max_diff >= tolerance:
            log.error(f"Backend mismatch {key}: max |diff| = {max_diff:.3e}.")
        else:
            log.test(f"Backend agreement {key} pass (max |diff| = {max_diff:.3e}).")
    return diffs


def plot_gain_vs_ncell(
    S_cells: np.ndarray,
    port_out: int | None = None,
    port_in: int = 0,
    freq_idx: int = 45,
    backend: Backend | str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot |S[port_out, port_in]| (dB) vs cumulative cell count, at a fixed
    frequency index, by replaying the sequential Redheffer-star cascade
    one cell at a time.

    Phase-matched parametric gain should grow ~linearly in dB with Nc (or
    track the coupled-mode-theory prediction at small epsilon, where pump
    depletion is negligible). A bend-over or saturation in this curve --
    cross-referenced against `check_cascade_conditioning` /
    `check_cascade_associativity` at the same cell count -- is the
    empirical signature of precision loss rather than a real physical
    effect.

    Parameters
    ----------
    S_cells : ndarray, shape (Nf, Nc, N, N) -- per-cell S-matrices, ordered
        left (input) to right (output).
    port_out : output port index; defaults to N // 2 (first right-side
        port), the usual transmission/gain element.
    port_in : input port index; defaults to 0 (first left-side port).
    freq_idx : which frequency slice (of Nf) to plot.
    ax : existing Axes to draw on; a new figure is created if None.
    """
    from numerical_solver.s_matrix import redheffer_star

    _, Nc, N, _ = S_cells.shape
    if port_out is None:
        port_out = 2

    total = S_cells[:, 0]
    gain_db = np.empty(Nc)
    gain_db[0] = 20.0 * np.log10(np.abs(total[freq_idx, port_out, port_in]))
    for c in range(1, Nc):
        total = redheffer_star(S_cells[:, c], total, backend=backend)
        gain_db[c] = 20.0 * np.log10(np.abs(total[freq_idx, port_out, port_in]))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, Nc + 1), gain_db, marker=".")
    ax.set_xlabel("Number of cascaded cells")
    ax.set_ylabel(f"|S[{port_out}, {port_in}]| (dB)")
    ax.grid(True, alpha=0.3)
    return ax


def plot_s_vs_frequency_at_cell(
    S_cells: np.ndarray,
    freqs: np.ndarray,
    port_out: int,
    port_in: int,
    cell_indices: int | list[int],
    backend: Backend | str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot |S[port_out, port_in]| (dB) vs frequency, cascading only the
    first `cell_idx` cells for each `cell_idx` in `cell_indices`.

    The frequency-domain counterpart to `plot_gain_vs_ncell` (which fixes
    frequency and sweeps cell count): this fixes cell count and sweeps
    frequency, so overlaying a few `cell_indices` shows how the frequency
    response builds up as cells are added -- e.g. whether a feature is
    already present at Nc/3 cells or only emerges near the full cascade.

    Parameters
    ----------
    S_cells : ndarray, shape (Nf, Nc, N, N) -- per-cell S-matrices, ordered
        left (input) to right (output).
    cell_indices : one or more cell counts (1..Nc) to cascade up to and
        plot as separate curves, e.g. [Nc // 2, Nc].
    """
    from numerical_solver.s_matrix import cascade_all

    if isinstance(cell_indices, int):
        cell_indices = [cell_indices]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for cell_idx in cell_indices:
        S_partial = cascade_all(S_cells[:, :cell_idx], backend=backend)
        db = 20.0 * np.log10(np.abs(S_partial[:, port_out, port_in]))
        ax.plot(freqs, db, label=f"cells={cell_idx}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(f"|S[{port_out}, {port_in}]| (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax
