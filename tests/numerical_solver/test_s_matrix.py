import numpy as np
import pytest

from numerical_solver.s_matrix import ABCD_to_S, SMatrix, redheffer_star


def _make_identity_S(Nf, N):
    """Through-connection S-matrix — identity element of the Redheffer product."""
    k = N // 2
    S = np.zeros((Nf, N, N), dtype=complex)
    S[:, :k, k:] = np.eye(k)
    S[:, k:, :k] = np.eye(k)
    return S


def _random_S(rng, Nf, N):
    return rng.standard_normal((Nf, N, N)) + 1j * rng.standard_normal((Nf, N, N))


def _random_ABCD(rng, N):
    """Random ABCD with well-conditioned lower-left (C) block."""
    M = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    k = N // 2
    M[k:, :k] += 2.0 * np.eye(k)
    return M


class TestRedhefferStar:
    def test_shape_4x4(self):
        rng = np.random.default_rng(0)
        S1, S2 = _random_S(rng, 10, 4), _random_S(rng, 10, 4)
        assert redheffer_star(S2, S1).shape == (10, 4, 4)

    def test_shape_6x6(self):
        rng = np.random.default_rng(1)
        S1, S2 = _random_S(rng, 5, 6), _random_S(rng, 5, 6)
        assert redheffer_star(S2, S1).shape == (5, 6, 6)

    def test_shape_2x2(self):
        rng = np.random.default_rng(2)
        S1, S2 = _random_S(rng, 8, 2), _random_S(rng, 8, 2)
        assert redheffer_star(S2, S1).shape == (8, 2, 2)

    def test_dtype_complex(self):
        rng = np.random.default_rng(3)
        S1, S2 = _random_S(rng, 4, 4), _random_S(rng, 4, 4)
        assert np.issubdtype(redheffer_star(S2, S1).dtype, np.complexfloating)

    def test_identity_right(self):
        """redheffer_star(S, S_id) == S."""
        rng = np.random.default_rng(4)
        S = _random_S(rng, 7, 4)
        S_id = _make_identity_S(7, 4)
        np.testing.assert_allclose(redheffer_star(S, S_id), S, atol=1e-12)

    def test_identity_left(self):
        """redheffer_star(S_id, S) == S."""
        rng = np.random.default_rng(5)
        S = _random_S(rng, 7, 4)
        S_id = _make_identity_S(7, 4)
        np.testing.assert_allclose(redheffer_star(S_id, S), S, atol=1e-12)

    def test_identity_6x6(self):
        rng = np.random.default_rng(6)
        S = _random_S(rng, 3, 6)
        S_id = _make_identity_S(3, 6)
        np.testing.assert_allclose(redheffer_star(S, S_id), S, atol=1e-12)
        np.testing.assert_allclose(redheffer_star(S_id, S), S, atol=1e-12)

    def test_associativity_4x4(self):
        """(A ⋆ B) ⋆ C == A ⋆ (B ⋆ C)."""
        rng = np.random.default_rng(7)
        A, B, C = [_random_S(rng, 6, 4) for _ in range(3)]
        lhs = redheffer_star(redheffer_star(C, B), A)
        rhs = redheffer_star(C, redheffer_star(B, A))
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_associativity_6x6(self):
        rng = np.random.default_rng(8)
        A, B, C = [_random_S(rng, 4, 6) for _ in range(3)]
        lhs = redheffer_star(redheffer_star(C, B), A)
        rhs = redheffer_star(C, redheffer_star(B, A))
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)


class TestABCDtoS:
    def test_shape_single_4x4(self):
        rng = np.random.default_rng(10)
        assert ABCD_to_S(_random_ABCD(rng, 4), 50.0).shape == (4, 4)

    def test_shape_single_2x2(self):
        rng = np.random.default_rng(11)
        assert ABCD_to_S(_random_ABCD(rng, 2), 50.0).shape == (2, 2)

    def test_shape_batched_4x4(self):
        rng = np.random.default_rng(12)
        Tb = np.stack([_random_ABCD(rng, 4) for _ in range(10)])
        assert ABCD_to_S(Tb, 50.0).shape == (10, 4, 4)

    def test_shape_batched_6x6(self):
        rng = np.random.default_rng(13)
        Tb = np.stack([_random_ABCD(rng, 6) for _ in range(5)])
        assert ABCD_to_S(Tb, 50.0).shape == (5, 6, 6)

    def test_dtype_complex(self):
        rng = np.random.default_rng(14)
        assert np.issubdtype(
            ABCD_to_S(_random_ABCD(rng, 4), 50.0).dtype, np.complexfloating
        )

    def test_known_2port_result(self):
        """ABCD = [[2, 3], [1, 2]], Z0=1 → S = 0.25 * ones(2,2).

        Verified against the standard 2-port ABCD→S formula:
          Δ = A + B/Z0 + C*Z0 + D = 2+3+1+2 = 8
          S11 = (2+3-1-2)/8 = 0.25,  S21 = 2/8 = 0.25, etc.
        """
        T = np.array([[2.0, 3.0], [1.0, 2.0]], dtype=complex)
        np.testing.assert_allclose(
            ABCD_to_S(T, 1.0), 0.25 * np.ones((2, 2)), atol=1e-12
        )

    def test_batched_matches_single(self):
        rng = np.random.default_rng(15)
        Tb = np.stack([_random_ABCD(rng, 4) for _ in range(8)])
        S_batch = ABCD_to_S(Tb, 50.0)
        for i in range(8):
            np.testing.assert_allclose(S_batch[i], ABCD_to_S(Tb[i], 50.0), atol=1e-12)

    def test_z0_scaling_changes_result(self):
        rng = np.random.default_rng(16)
        T = _random_ABCD(rng, 4)
        assert not np.allclose(ABCD_to_S(T, 50.0), ABCD_to_S(T, 100.0))


class TestSMatrix:
    def test_init_single_adds_batch_dim(self):
        rng = np.random.default_rng(20)
        data = _random_S(rng, 1, 4)[0]  # (4, 4)
        sm = SMatrix(data, Z0=50.0)
        assert sm.shape == (1, 4, 4)

    def test_init_batched_preserved(self):
        rng = np.random.default_rng(21)
        data = _random_S(rng, 7, 4)
        sm = SMatrix(data, Z0=50.0)
        assert sm.shape == (7, 4, 4)

    def test_init_invalid_odd_N_raises(self):
        with pytest.raises(ValueError):
            SMatrix(np.zeros((3, 3), dtype=complex))

    def test_init_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            SMatrix(np.zeros((2, 4, 4, 4), dtype=complex))

    def test_from_ABCD_single(self):
        rng = np.random.default_rng(22)
        T = _random_ABCD(rng, 4)
        sm = SMatrix.from_ABCD(T, Z0=50.0)
        assert sm.shape == (1, 4, 4)
        assert np.issubdtype(sm.array.dtype, np.complexfloating)

    def test_from_ABCD_batched(self):
        rng = np.random.default_rng(23)
        Tb = np.stack([_random_ABCD(rng, 4) for _ in range(5)])
        sm = SMatrix.from_ABCD(Tb, Z0=50.0)
        assert sm.shape == (5, 4, 4)

    def test_from_ABCD_matches_pure_function(self):
        rng = np.random.default_rng(24)
        T = _random_ABCD(rng, 4)
        sm = SMatrix.from_ABCD(T, Z0=50.0)
        np.testing.assert_allclose(sm.array[0], ABCD_to_S(T, 50.0), atol=1e-14)

    def test_from_ABCD_known_result(self):
        """from_ABCD with the known 2-port case → S entries all 0.25."""
        T = np.array([[2.0, 3.0], [1.0, 2.0]], dtype=complex)
        sm = SMatrix.from_ABCD(T, Z0=1.0)
        np.testing.assert_allclose(sm.array[0], 0.25 * np.ones((2, 2)), atol=1e-12)

    def test_Nf_and_N(self):
        rng = np.random.default_rng(25)
        sm = SMatrix(_random_S(rng, 9, 6), Z0=75.0)
        assert sm.Nf == 9
        assert sm.N == 6

    def test_Z0_stored(self):
        rng = np.random.default_rng(26)
        sm = SMatrix(_random_S(rng, 3, 4), Z0=75.0)
        assert sm.Z0 == 75.0

    def test_array_returns_data(self):
        rng = np.random.default_rng(27)
        data = _random_S(rng, 4, 4)
        sm = SMatrix(data, Z0=50.0)
        np.testing.assert_array_equal(sm.array, data)

    def test_block_shapes(self):
        rng = np.random.default_rng(28)
        sm = SMatrix(_random_S(rng, 5, 4), Z0=50.0)
        for block in (sm.S11, sm.S12, sm.S21, sm.S22):
            assert block.shape == (5, 2, 2)

    def test_block_values_match_slices(self):
        rng = np.random.default_rng(29)
        data = _random_S(rng, 3, 4)
        sm = SMatrix(data, Z0=50.0)
        np.testing.assert_array_equal(sm.S11, data[:, :2, :2])
        np.testing.assert_array_equal(sm.S12, data[:, :2, 2:])
        np.testing.assert_array_equal(sm.S21, data[:, 2:, :2])
        np.testing.assert_array_equal(sm.S22, data[:, 2:, 2:])

    def test_repr(self):
        rng = np.random.default_rng(30)
        sm = SMatrix(_random_S(rng, 3, 4), Z0=50.0)
        assert "SMatrix" in repr(sm)
        assert "50.0" in repr(sm)

    def test_getitem_returns_smatrix(self):
        rng = np.random.default_rng(31)
        sm = SMatrix(_random_S(rng, 6, 4), Z0=50.0)
        assert isinstance(sm[0], SMatrix)

    def test_getitem_single_index_gives_nf1(self):
        rng = np.random.default_rng(32)
        sm = SMatrix(_random_S(rng, 6, 4), Z0=50.0)
        assert sm[0].Nf == 1
        assert sm[0].N == 4

    def test_getitem_slice(self):
        rng = np.random.default_rng(33)
        sm = SMatrix(_random_S(rng, 6, 4), Z0=50.0)
        assert sm[1:4].shape == (3, 4, 4)

    def test_getitem_preserves_Z0(self):
        rng = np.random.default_rng(34)
        sm = SMatrix(_random_S(rng, 6, 4), Z0=75.0)
        assert sm[2].Z0 == 75.0

    def test_matmul_returns_smatrix(self):
        rng = np.random.default_rng(40)
        sm1 = SMatrix(_random_S(rng, 5, 4), Z0=50.0)
        sm2 = SMatrix(_random_S(rng, 5, 4), Z0=50.0)
        assert isinstance(sm1 @ sm2, SMatrix)

    def test_matmul_shape(self):
        rng = np.random.default_rng(41)
        sm1 = SMatrix(_random_S(rng, 5, 4), Z0=50.0)
        sm2 = SMatrix(_random_S(rng, 5, 4), Z0=50.0)
        assert (sm1 @ sm2).shape == (5, 4, 4)

    def test_matmul_identity_right(self):
        """sm @ sm_id == sm."""
        rng = np.random.default_rng(42)
        sm = SMatrix(_random_S(rng, 5, 4), Z0=50.0)
        sm_id = SMatrix(_make_identity_S(5, 4), Z0=50.0)
        np.testing.assert_allclose((sm @ sm_id).array, sm.array, atol=1e-12)

    def test_matmul_identity_left(self):
        """sm_id @ sm == sm."""
        rng = np.random.default_rng(43)
        sm = SMatrix(_random_S(rng, 5, 4), Z0=50.0)
        sm_id = SMatrix(_make_identity_S(5, 4), Z0=50.0)
        np.testing.assert_allclose((sm_id @ sm).array, sm.array, atol=1e-12)

    def test_matmul_associativity(self):
        """(A @ B) @ C == A @ (B @ C)."""
        rng = np.random.default_rng(44)
        A = SMatrix(_random_S(rng, 4, 4), Z0=50.0)
        B = SMatrix(_random_S(rng, 4, 4), Z0=50.0)
        C = SMatrix(_random_S(rng, 4, 4), Z0=50.0)
        np.testing.assert_allclose(((A @ B) @ C).array, (A @ (B @ C)).array, atol=1e-10)

    def test_matmul_matches_pure_function(self):
        rng = np.random.default_rng(45)
        d1 = _random_S(rng, 5, 4)
        d2 = _random_S(rng, 5, 4)
        sm1 = SMatrix(d1, Z0=50.0)
        sm2 = SMatrix(d2, Z0=50.0)
        expected = redheffer_star(d2, d1)
        np.testing.assert_allclose((sm1 @ sm2).array, expected, atol=1e-14)

    def test_matmul_z0_mismatch_raises(self):
        rng = np.random.default_rng(46)
        sm1 = SMatrix(_random_S(rng, 3, 4), Z0=50.0)
        sm2 = SMatrix(_random_S(rng, 3, 4), Z0=75.0)
        with pytest.raises(ValueError, match="Z0"):
            sm1 @ sm2

    def test_matmul_N_mismatch_raises(self):
        rng = np.random.default_rng(47)
        sm1 = SMatrix(_random_S(rng, 3, 4), Z0=50.0)
        sm2 = SMatrix(_random_S(rng, 3, 6), Z0=50.0)
        with pytest.raises(ValueError):
            sm1 @ sm2
