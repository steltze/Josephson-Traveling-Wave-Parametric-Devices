import numpy as np
import pytest

from backends.numpy_backend import NumpyBackend
from numerical_solver.s_matrix import ABCD_to_S, redheffer_star


def _make_identity_S(Nf, N):
    """Through-connection S-matrix — identity element of the Redheffer product."""
    k = N // 2
    S = np.zeros((Nf, N, N), dtype=complex)
    S[:, :k, k:] = np.eye(k)
    S[:, k:, :k] = np.eye(k)
    return S


class TestNumpyBackendDirect:
    def test_abcd_to_s_known_2port_result(self):
        """Same fixture as the module-level ABCD_to_S test, called on the backend directly."""
        backend = NumpyBackend()
        abcd = np.array([[[2.0, 3.0], [1.0, 2.0]]], dtype=complex)
        z0 = np.full((1, 2), 1.0, dtype=complex)
        np.testing.assert_allclose(
            backend.abcd_to_s(abcd, z0), 0.25 * np.ones((1, 2, 2)), atol=1e-12
        )

    def test_redheffer_star_identity(self):
        backend = NumpyBackend()
        rng = np.random.default_rng(0)
        S = rng.standard_normal((5, 4, 4)) + 1j * rng.standard_normal((5, 4, 4))
        S_id = _make_identity_S(5, 4)
        np.testing.assert_allclose(backend.redheffer_star(S_id, S), S, atol=1e-12)


class TestBackendParamOnPublicFunctions:
    """`ABCD_to_S` / `redheffer_star` accept `backend` and match the default path."""

    def test_abcd_to_s_explicit_instance_matches_default(self):
        rng = np.random.default_rng(1)
        T = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        T[2:, :2] += 2.0 * np.eye(2)
        default = ABCD_to_S(T, 50.0)
        explicit = ABCD_to_S(T, 50.0, backend=NumpyBackend())
        np.testing.assert_allclose(default, explicit, atol=1e-14)

    def test_abcd_to_s_name_string_matches_default(self):
        rng = np.random.default_rng(2)
        T = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        T[2:, :2] += 2.0 * np.eye(2)
        np.testing.assert_allclose(
            ABCD_to_S(T, 50.0), ABCD_to_S(T, 50.0, backend="numpy"), atol=1e-14
        )

    def test_redheffer_star_name_string_matches_default(self):
        rng = np.random.default_rng(3)
        S1 = rng.standard_normal((3, 4, 4)) + 1j * rng.standard_normal((3, 4, 4))
        S2 = rng.standard_normal((3, 4, 4)) + 1j * rng.standard_normal((3, 4, 4))
        np.testing.assert_allclose(
            redheffer_star(S2, S1), redheffer_star(S2, S1, backend="numpy"), atol=1e-14
        )

    def test_unknown_backend_name_raises(self):
        rng = np.random.default_rng(4)
        T = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        T[2:, :2] += 2.0 * np.eye(2)
        with pytest.raises(ValueError, match="Unknown backend"):
            ABCD_to_S(T, 50.0, backend="does-not-exist")
