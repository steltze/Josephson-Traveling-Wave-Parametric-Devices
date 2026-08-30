import numpy as np

from analysis.dispersion_relation import bloch_wavenumbers


def _diagonal_T(gammas: np.ndarray) -> np.ndarray:
    """(Nf, N, N) diagonal transfer matrix with eigenvalues exp(-gammas)."""
    Nf, N = gammas.shape
    T = np.zeros((Nf, N, N), dtype=complex)
    idx = np.arange(N)
    T[:, idx, idx] = np.exp(-gammas)
    return T


class TestBlochWavenumbers:
    def test_recovers_known_gamma_for_diagonal_cell(self):
        """For a diagonal T (already the eigenbasis), alpha/k must reproduce
        the real/imag parts of gamma = -log(eigenvalue), sorted descending
        in k."""
        gammas = np.array([[0.1 + 0.5j, 0.2 - 0.5j, 0.05 + 0.0j]])  # (Nf=1, N=3)
        T = _diagonal_T(gammas)
        alpha, k = bloch_wavenumbers(T)

        # Sorted descending by k (imag part of gamma).
        assert np.all(np.diff(k[0]) <= 0)
        np.testing.assert_allclose(sorted(alpha[0]), sorted(gammas[0].real), atol=1e-10)
        np.testing.assert_allclose(sorted(k[0]), sorted(gammas[0].imag), atol=1e-10)

    def test_lossless_identity_cell_has_zero_attenuation(self):
        """A lossless through-connection (T = I) must have alpha == 0 and
        k == 0 for every mode -- no propagation, no loss."""
        T = np.tile(np.eye(2, dtype=complex), (4, 1, 1))
        alpha, k = bloch_wavenumbers(T)
        np.testing.assert_allclose(alpha, 0.0, atol=1e-12)
        np.testing.assert_allclose(k, 0.0, atol=1e-12)

    def test_two_port_matches_scalar_case(self):
        """For N=2, the two Bloch modes should be +/-k of each other for a
        lossless propagating cell (eigenvalues on the unit circle)."""
        theta = 0.7
        T = np.zeros((1, 2, 2), dtype=complex)
        T[0, 0, 0] = np.exp(-1j * theta)
        T[0, 1, 1] = np.exp(1j * theta)
        alpha, k = bloch_wavenumbers(T)
        np.testing.assert_allclose(alpha, 0.0, atol=1e-12)
        np.testing.assert_allclose(sorted(k[0]), sorted([-theta, theta]), atol=1e-10)

    def test_return_eigenvectors_shape_and_consistency(self):
        rng = np.random.default_rng(7)
        Nf, N = 3, 4
        data = rng.standard_normal((Nf, N, N)) + 1j * rng.standard_normal((Nf, N, N))
        # Make it diagonalizable/well-conditioned by biasing the diagonal.
        data += 3.0 * np.eye(N)[None, :, :]

        alpha, k = bloch_wavenumbers(data)
        alpha_ev, k_ev, eigvecs = bloch_wavenumbers(data, return_eigenvectors=True)

        np.testing.assert_allclose(alpha, alpha_ev)
        np.testing.assert_allclose(k, k_ev)

        # T @ v_i == eigenvalue_i * v_i for each reordered eigenvector -- the
        # returned eigenvectors must stay paired with alpha/k after the
        # descending-k reordering, not just be *some* eigenbasis of T.
        eigenvalues = np.exp(-(alpha_ev + 1j * k_ev))  # (Nf, N)
        lhs = np.einsum("fij,fjk->fik", data, eigvecs)
        rhs = eigvecs * eigenvalues[:, None, :]
        np.testing.assert_allclose(lhs, rhs, atol=1e-8)
