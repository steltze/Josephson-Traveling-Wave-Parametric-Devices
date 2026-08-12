import os
import sys

# Under pytest, the repo root (for `src.*` imports below) and `src/` (for the
# bare `simulation`/`models`/`numerical_solver` imports inside
# compare_manual_squid_vs_solver) are already on sys.path via pytest's
# rootdir insertion and the `pythonpath` setting in pyproject.toml. Running
# this file directly (`python tests/symbolic_solver/test_cell_single_mode_symmetric.py`)
# gets neither, so add them explicitly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from src.symbolic_solver.cell_single_mode_symmetric import CellSingleModeSymmetric
from src.symbolic_solver.cell_single_mode import CellSingleMode
from src.models.cell import CellImmitance

_OMEGA_S = 1.0
_OMEGA_P = 0.1
_THETA = np.pi / 4
_YG0_0 = 0.5 + 0.0j
_YG0_1 = 0.5 + 0.0j
_ZS0_0 = 1.0 + 0.0j
_ZS0_1 = 1.0 + 0.0j


def _omega_val(q):
    return _OMEGA_S + q * _OMEGA_P


def _Zs_num(m, omega):
    """Frequency-dependent m=1 harmonic so Zs1(omega[0]) != Zs1(omega[1]),
    which lets a test tell apart an omega-index mixup from a p/m mixup."""
    return {1: 0.10 + 0.05j + 0.05 * omega}.get(m, 0j)


def _Yg_num(m, omega):
    return {1: 0.05 + 0.02j}.get(m, 0j)


def _make_cell(theta=_THETA):
    return CellImmitance(
        theta=theta,
        Zs0_fn=lambda omega: _ZS0_0 if omega == _omega_val(0) else _ZS0_1,
        Yg0_fn=lambda omega: _YG0_0 if omega == _omega_val(0) else _YG0_1,
        Zs_harm_fn=_Zs_num,
        Yg_harm_fn=_Yg_num,
    )


def _expected_symmetric_T(Zs0_0, Zs0_1, Zs1p_0, Zs1m_1, Yg0_0, Yg0_1, theta):
    """
    Hand-derived 4x4 transfer matrix for the symmetric Pi cell (M=1,
    ks_state=[0, 1]), built directly from the cell equations

        Ia          = I(t) - (Yg(t)/2) * V(t)
        V(t)_next   = V(t) - Zs(t) * Ia
        I(t)_next   = Ia - (Yg(t)/2) * V(t)_next

    without going through any of CellSingleModeSymmetric's machinery. Yg(t)
    currently carries no harmonic content (see
    CellSingleModeSymmetric._build_Yg_shunt), so Yg0_k multiplies V[k]/I[k]
    with no sideband mixing; only Zs(t) couples k=0 and k=1 via its m=1
    Fourier terms. Zs1p(omega[0]) drives the k=0 -> k=1 (E^{+1}) coupling
    and Zs1m(omega[1]) drives k=1 -> k=0 (E^{-1}); Zs1p(omega[1]) and
    Zs1m(omega[0]) would couple to k=2/k=-1 and are truncated away.

    This matches the analytic matrix in transfer_matrix.pdf (rendered from
    this same cell via export_matrix_graphic).
    """
    ep = np.exp(1j * theta)
    em = np.exp(-1j * theta)
    return np.array(
        [
            [Yg0_0 * Zs0_0 / 2.0 + 1, Yg0_1 * Zs1m_1 * ep / 2.0, -Zs0_0, -Zs1m_1 * ep],
            [Yg0_0 * Zs1p_0 * em / 2.0, Yg0_1 * Zs0_1 / 2.0 + 1, -Zs1p_0 * em, -Zs0_1],
            [
                -(Yg0_0**2) * Zs0_0 / 4.0 - Yg0_0,
                -Yg0_0 * Yg0_1 * Zs1m_1 * ep / 4.0,
                Yg0_0 * Zs0_0 / 2.0 + 1,
                Yg0_0 * Zs1m_1 * ep / 2.0,
            ],
            [
                -Yg0_0 * Yg0_1 * Zs1p_0 * em / 4.0,
                -(Yg0_1**2) * Zs0_1 / 4.0 - Yg0_1,
                Yg0_1 * Zs1p_0 * em / 2.0,
                Yg0_1 * Zs0_1 / 2.0 + 1,
            ],
        ],
        dtype=complex,
    )


@pytest.fixture(scope="module")
def T_num_symmetric():
    cell_obj = CellSingleModeSymmetric()
    M, ks_state = 1, [0, 1]
    T_sym, state_syms, Zs_m_p, Zs_m_m, Yg_m_p, Yg_m_m = (
        cell_obj.build_symbolic_transfer_matrix(M, ks_state)
    )
    dim = len(state_syms)
    return cell_obj.build_numeric_matrix(
        T_sym,
        dim,
        M,
        ks_state,
        Zs_m_p,
        Zs_m_m,
        Yg_m_p,
        Yg_m_m,
        _make_cell(),
        _omega_val,
        _OMEGA_P,
        k_val=0,
    )


@pytest.fixture(scope="module")
def T_expected():
    w0, w1 = _omega_val(0), _omega_val(1)
    Zs1p_0 = _Zs_num(1, w0)  # E^{+1} term, evaluated at the source sideband omega[0]
    Zs1m_1 = _Zs_num(1, w1)  # E^{-1} term, evaluated at the source sideband omega[1]
    return _expected_symmetric_T(_ZS0_0, _ZS0_1, Zs1p_0, Zs1m_1, _YG0_0, _YG0_1, _THETA)


class TestManualMatrixMatchesSolver:
    """Substitute numbers into the hand-derived Pi-cell formula and check
    every entry of CellSingleModeSymmetric's numerical output against it."""

    def test_shape(self, T_num_symmetric):
        assert T_num_symmetric.shape == (4, 4)

    def test_full_matrix_matches_manual_derivation(self, T_num_symmetric, T_expected):
        np.testing.assert_allclose(T_num_symmetric, T_expected, atol=1e-10)

    @pytest.mark.parametrize("i,j", [(i, j) for i in range(4) for j in range(4)])
    def test_each_entry(self, T_num_symmetric, T_expected, i, j):
        assert T_num_symmetric[i, j] == pytest.approx(T_expected[i, j], abs=1e-10)

    def test_voltage_diagonal_has_self_shunt_term(self, T_num_symmetric):
        """Pi cell's V-rows are NOT a bare identity: T[0,0]=T[1,1]=Yg0*Zs0/2+1,
        the signature of the half-shunt folded into each voltage update."""
        expected_diag = _YG0_0 * _ZS0_0 / 2 + 1
        assert T_num_symmetric[0, 0] == pytest.approx(expected_diag, abs=1e-10)
        assert T_num_symmetric[1, 1] == pytest.approx(expected_diag, abs=1e-10)
        assert T_num_symmetric[0, 0] == pytest.approx(T_num_symmetric[1, 1], abs=1e-10)

    def test_cross_coupling_uses_source_sideband_not_target(self, T_num_symmetric):
        """T[0,1] must use Zs1(omega[1]) and T[1,0] must use Zs1(omega[0]);
        a solver bug that swaps these would still pass a same-valued-Zs1
        test but fails here since Zs1 is frequency-dependent."""
        w0, w1 = _omega_val(0), _omega_val(1)
        Zs1p_0 = _Zs_num(1, w0)
        Zs1m_1 = _Zs_num(1, w1)
        assert Zs1p_0 != Zs1m_1  # sanity: the two harmonics really do differ
        assert T_num_symmetric[1, 0] == pytest.approx(
            _YG0_0 * Zs1p_0 * np.exp(-1j * _THETA) / 2, abs=1e-10
        )
        assert T_num_symmetric[0, 1] == pytest.approx(
            _YG0_1 * Zs1m_1 * np.exp(1j * _THETA) / 2, abs=1e-10
        )

    def test_determinant_is_one(self, T_num_symmetric):
        """Reciprocal, lossless cell: |det(T)| == 1 per cell."""
        det = np.linalg.det(T_num_symmetric)
        assert abs(det) == pytest.approx(1.0, abs=1e-8)


@pytest.fixture(scope="module")
def T_num_L_cell():
    """Same underlying Zs0/Yg0/Zs1 immitances, but through the plain L
    (series-then-shunt) cell instead of the symmetric Pi cell."""
    cell_obj = CellSingleMode()
    M, ks_state = 1, [0, 1]
    T_sym, state_syms, Zs_m_p, Zs_m_m, Yg_m_p, Yg_m_m = (
        cell_obj.build_symbolic_transfer_matrix(M, ks_state)
    )
    dim = len(state_syms)
    return cell_obj.build_numeric_matrix(
        T_sym,
        dim,
        M,
        ks_state,
        Zs_m_p,
        Zs_m_m,
        Yg_m_p,
        Yg_m_m,
        _make_cell(),
        _omega_val,
        _OMEGA_P,
        k_val=0,
    )


class TestCompareAgainstLCell:
    """Cross-topology sanity checks in the spirit of
    examples/compare_cell_topologies.py: both cells must independently be
    lossless/reciprocal (|det|=1) for the same immitances, while differing
    in exactly the way their circuit topologies predict."""

    def test_both_topologies_have_unit_determinant(self, T_num_symmetric, T_num_L_cell):
        assert abs(np.linalg.det(T_num_symmetric)) == pytest.approx(1.0, abs=1e-8)
        assert abs(np.linalg.det(T_num_L_cell)) == pytest.approx(1.0, abs=1e-8)

    def test_L_cell_voltage_block_is_bare_identity(self, T_num_L_cell):
        """The plain L cell has no shunt folded into its V update: T[0,0]=T[1,1]=1
        exactly, unlike the symmetric Pi cell's Yg0*Zs0/2+1."""
        assert T_num_L_cell[0, 0] == pytest.approx(1.0, abs=1e-12)
        assert T_num_L_cell[1, 1] == pytest.approx(1.0, abs=1e-12)

    def test_topologies_disagree_on_voltage_diagonal(
        self, T_num_symmetric, T_num_L_cell
    ):
        """This is the numeric signature that distinguishes the two topologies:
        the Pi cell's extra half-shunt term makes T[0,0] != 1."""
        assert T_num_symmetric[0, 0] != pytest.approx(T_num_L_cell[0, 0], abs=1e-6)

    def test_series_impedance_coupling_matches_between_topologies(
        self, T_num_symmetric, T_num_L_cell
    ):
        """Both topologies route -Zs0(omega[k]) directly from I[k,n] to
        V[k,n+1] on the diagonal-in-sideband block, independent of the
        shunt placement."""
        assert T_num_symmetric[0, 2] == pytest.approx(T_num_L_cell[0, 2], abs=1e-10)
        assert T_num_symmetric[1, 3] == pytest.approx(T_num_L_cell[1, 3], abs=1e-10)


def prepare_manual_ABCDs_symmetric(cfg):
    """
    Vectorized (Nf, Nc, 4, 4) per-cell ABCD matrices for the symmetric Pi
    cell, built directly from the closed-form formula in
    `_expected_symmetric_T` -- the same one hand-verified against
    transfer_matrix.pdf above -- using the real squid
    (ModulatedInductor || junction self-capacitance) component that
    `JTLDiscrete.build` uses for the "pi" topology.

    This bypasses CellSingleModeSymmetric's sympy pipeline entirely: if
    `compare_manual_squid_vs_solver`'s two |S31| curves disagree, the bug
    is in CellSingleModeSymmetric's symbolic derivation or its numeric
    substitution (build_symbolic_transfer_matrix / build_cell_freq_matrices),
    not in the underlying cell physics.
    """
    from models.electrical_elements import Component, ModulatedInductor, Capacitor

    ncell = cfg.ncell
    ns = np.arange(ncell)
    a = cfg.cell_size

    ZR = cfg.Z0 * np.ones(ncell)
    L = ZR / cfg.omega_cutoff * 2
    C = 2.0 / (cfg.omega_cutoff * ZR)
    Cs_jj = 1.0 / (cfg.omega_j**2 * L)
    wj = 1.0 / np.sqrt(L * Cs_jj)

    thetas = cfg.omega_pump / cfg.v_pump * ns * a

    w_s = np.asarray(cfg.omegas)  # (Nf,)
    w0, w1 = w_s, w_s + cfg.omega_pump
    omega_sb = np.stack([w0, w1], axis=1)  # (Nf, 2): sideband grid, same every cell

    Nf = len(w_s)
    ABCDs = np.empty((Nf, ncell, 4, 4), dtype=complex)

    for i in range(ncell):
        Ccap_val = 1.0 / (wj[i] ** 2 * L[i])
        squid = Component.parallel(
            ModulatedInductor(L0=L[i], eps=cfg.epsilon, order=cfg.M),
            Capacitor(Ccap_val),
        )
        Zs = squid.impedance(omega_sb)  # (Nf, 2, 2): [:, target, source]
        Zs0_0, Zs0_1 = Zs[:, 0, 0], Zs[:, 1, 1]
        Zs1p_0, Zs1m_1 = Zs[:, 1, 0], Zs[:, 0, 1]

        cap = Capacitor(C[i])
        Yg0_0, Yg0_1 = cap.admittance(w0), cap.admittance(w1)

        T_cell = _expected_symmetric_T(
            Zs0_0, Zs0_1, Zs1p_0, Zs1m_1, Yg0_0, Yg0_1, thetas[i]
        )  # (4, 4, Nf)
        ABCDs[:, i] = T_cell.transpose(2, 0, 1)

    return ABCDs


def compare_manual_squid_vs_solver():
    """
    Diagnostic script (not a pytest test -- run directly):

        python -m tests.symbolic_solver.test_cell_single_mode_symmetric

    Cascades the hand-built ABCD matrices from `prepare_manual_ABCDs_symmetric`
    the same way `Simulation.get_s_matrix` cascades its own (per-cell S via
    ABCD_to_S, then Redheffer-star via cascade_all), and overlays |S31| (dB)
    against `Simulation(JTLDiscrete, cfg, cell_topology="pi")`'s solver-driven
    result for the identical device. Agreement pinpoints any "doesn't work"
    symptom to cascade/config wiring rather than the symbolic solver;
    disagreement pinpoints it to CellSingleModeSymmetric itself.
    """
    import matplotlib.pyplot as plt
    from simulation import SimulationConfig, Simulation
    from models import JTLDiscrete
    from numerical_solver.s_matrix import ABCD_to_S, cascade_all

    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=[0, 1],
        ncell=320,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.045,
        omega_c=3.4 * 2 * np.pi,
        v_ratio=-2.5,
        freq_min=1,
        freq_max=12,
        n_freqs=500,
        disorder=False,
        epsilon_nramp=0,
    )
    signal_port = 0

    ABCD_manual = prepare_manual_ABCDs_symmetric(cfg)  # (Nf, Nc, 4, 4)
    Nf, Nc = ABCD_manual.shape[:2]
    S_manual_cells = ABCD_to_S(
        np.linalg.inv(ABCD_manual.reshape(Nf * Nc, 4, 4)), cfg.Z0
    ).reshape(Nf, Nc, 4, 4)
    S_manual = cascade_all(S_manual_cells)

    sim = Simulation(JTLDiscrete, cfg, cell_topology="pi")
    S_solver = sim.get_s_matrix(normalize=False).array

    s31_manual = 20 * np.log10(np.abs(S_manual[:, 2, signal_port]))
    s31_solver = 20 * np.log10(np.abs(S_solver[:, 2, signal_port]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(cfg.freqs, s31_manual, label="manual closed-form (squid)")
    ax1.plot(cfg.freqs, s31_solver, "--", label="CellSingleModeSymmetric solver")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("|S31| (dB)")
    ax1.set_title("Signal -> idler gain: manual vs solver")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(cfg.freqs, s31_manual - s31_solver)
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Delta |S31| (dB)")
    ax2.set_title("manual - solver")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_manual_squid_vs_solver()
