import numpy as np
import pytest

from backends.numpy_backend import NumpyBackend

numba = pytest.importorskip("numba")

from backends.numba_backend import NumbaBackend


def _random_abcd(rng, N):
    """Random ABCD with a well-conditioned lower-left (C) block."""
    M = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    k = N // 2
    M[k:, :k] += 2.0 * np.eye(k)
    return M


def _random_s(rng, Nf, N):
    return rng.standard_normal((Nf, N, N)) + 1j * rng.standard_normal((Nf, N, N))


def _make_identity_s(Nf, N):
    k = N // 2
    S = np.zeros((Nf, N, N), dtype=complex)
    S[:, :k, k:] = np.eye(k)
    S[:, k:, :k] = np.eye(k)
    return S


def _random_zy(rng, Nf, m):
    Zs = rng.standard_normal((Nf, m, m)) + 1j * rng.standard_normal((Nf, m, m))
    Yg = rng.standard_normal((Nf, m, m)) + 1j * rng.standard_normal((Nf, m, m))
    return Zs, Yg


class _StubCell:
    def __init__(self, Zs, Yg):
        self.Zs_harm_fn = Zs
        self.Yg_harm_fn = Yg


class TestNumbaMatchesNumpy:
    @pytest.mark.parametrize("m", [1, 2, 4])
    def test_single_mode_matrix(self, m):
        rng = np.random.default_rng(50 + m)
        Zs, Yg = _random_zy(rng, 5, m)
        expected = NumpyBackend().single_mode_matrix(Zs, Yg)
        actual = NumbaBackend().single_mode_matrix(Zs, Yg)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize("m", [1, 2, 4])
    def test_symmetric_single_mode_matrix(self, m):
        rng = np.random.default_rng(60 + m)
        Zs, Yg = _random_zy(rng, 5, m)
        expected = NumpyBackend().symmetric_single_mode_matrix(Zs, Yg)
        actual = NumbaBackend().symmetric_single_mode_matrix(Zs, Yg)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize("topology", ["L", "pi"])
    def test_single_mode_matrix_grid(self, topology):
        rng = np.random.default_rng(70)
        Nf, m, Ncells = 4, 3, 5
        cells = [_StubCell(*_random_zy(rng, Nf, m)) for _ in range(Ncells)]
        expected = NumpyBackend().single_mode_matrix_grid(cells, topology)
        actual = NumbaBackend().single_mode_matrix_grid(cells, topology)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_abcd_to_s(self, N):
        rng = np.random.default_rng(100 + N)
        abcd = np.stack([_random_abcd(rng, N) for _ in range(6)])
        z0 = np.full((6, N), 50.0, dtype=complex)

        expected = NumpyBackend().abcd_to_s(abcd, z0)
        actual = NumbaBackend().abcd_to_s(abcd, z0)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_abcd_to_s_nonuniform_z0(self, N):
        rng = np.random.default_rng(200 + N)
        abcd = np.stack([_random_abcd(rng, N) for _ in range(5)])
        z0 = 40.0 + 20.0 * rng.random((5, N))

        expected = NumpyBackend().abcd_to_s(abcd, z0)
        actual = NumbaBackend().abcd_to_s(abcd, z0)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_redheffer_star(self, N):
        rng = np.random.default_rng(300 + N)
        s1 = _random_s(rng, 7, N)
        s2 = _random_s(rng, 7, N)

        expected = NumpyBackend().redheffer_star(s2, s1)
        actual = NumbaBackend().redheffer_star(s2, s1)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_redheffer_star_identity(self):
        rng = np.random.default_rng(400)
        S = _random_s(rng, 5, 4)
        S_id = _make_identity_s(5, 4)
        np.testing.assert_allclose(
            NumbaBackend().redheffer_star(S_id, S), S, atol=1e-10
        )

    @pytest.mark.parametrize("N", [2, 4, 6])
    @pytest.mark.parametrize("Nc", [1, 2, 5])
    def test_cascade_all_matches_numpy(self, N, Nc):
        """The fused numba cascade_all matches NumpyBackend's default reduce."""
        rng = np.random.default_rng(600 + N * 10 + Nc)
        Nf = 4
        s_cells = rng.standard_normal((Nf, Nc, N, N)) + 1j * rng.standard_normal(
            (Nf, Nc, N, N)
        )

        expected = NumpyBackend().cascade_all(s_cells)
        actual = NumbaBackend().cascade_all(s_cells)
        np.testing.assert_allclose(actual, expected, atol=1e-9)


class TestNumbaViaRegistry:
    def test_get_backend_numba(self):
        from backends import get_backend

        assert isinstance(get_backend("numba"), NumbaBackend)

    def test_public_functions_accept_numba_backend(self):
        from numerical_solver.s_matrix import ABCD_to_S

        rng = np.random.default_rng(500)
        T = _random_abcd(rng, 4)
        default = ABCD_to_S(T, 50.0)
        via_numba = ABCD_to_S(T, 50.0, backend="numba")
        np.testing.assert_allclose(default, via_numba, atol=1e-10)

    def test_cascade_all_accepts_numba_backend(self):
        from numerical_solver.s_matrix import cascade_all

        rng = np.random.default_rng(700)
        Nf, Nc, N = 3, 4, 4
        s_cells = rng.standard_normal((Nf, Nc, N, N)) + 1j * rng.standard_normal(
            (Nf, Nc, N, N)
        )
        default = cascade_all(s_cells)
        via_numba = cascade_all(s_cells, backend="numba")
        np.testing.assert_allclose(default, via_numba, atol=1e-9)
