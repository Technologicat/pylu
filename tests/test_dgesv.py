"""Tests for PyLU dense linear solver."""

import numpy as np
import pytest

import pylu


class TestSolve:
    """Test pylu.solve() — combined LU + solve."""

    def test_basic_solve(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        b = rng.random(5)
        x = pylu.solve(A, b)
        np.testing.assert_allclose(A @ x, b, rtol=1e-12)

    def test_identity(self):
        A = np.eye(4)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = pylu.solve(A, b)
        np.testing.assert_allclose(x, b, rtol=1e-15)

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 50])
    def test_various_sizes(self, n):
        rng = np.random.default_rng(123 + n)
        A = rng.random((n, n))
        b = rng.random(n)
        x = pylu.solve(A, b)
        np.testing.assert_allclose(A @ x, b, rtol=1e-10)

    def test_singular_raises(self):
        A = np.array([[1.0, 0.0], [0.0, 0.0]])
        b = np.array([1.0, 1.0])
        with pytest.raises(RuntimeError):
            pylu.solve(A, b)


class TestLUP:
    """Test lup() — human-readable L, U, p decomposition."""

    def test_lup_roundtrip(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        L, U, p = pylu.lup(A)
        # P A = L U, i.e. A[p, :] = L @ U
        np.testing.assert_allclose(A[p, :], L @ U, rtol=1e-12)

    def test_l_is_lower_triangular_with_unit_diag(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        L, U, p = pylu.lup(A)
        np.testing.assert_allclose(np.diag(L), np.ones(5))
        assert np.allclose(L, np.tril(L))

    def test_u_is_upper_triangular(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        L, U, p = pylu.lup(A)
        assert np.allclose(U, np.triu(U))


class TestFactorizeThenSolve:
    """Test lup_packed() + solve_decomposed() workflow."""

    def test_basic(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        b = rng.random(5)
        LU, p = pylu.lup_packed(A)
        x = pylu.solve_decomposed(LU, p, b)
        np.testing.assert_allclose(A @ x, b, rtol=1e-12)

    def test_many_rhs_same_lhs(self):
        """The main use case: factorize once, solve with many RHS vectors."""
        rng = np.random.default_rng(42)
        n = 8
        A = rng.random((n, n))
        LU, p = pylu.lup_packed(A)

        for _ in range(20):
            b = rng.random(n)
            x = pylu.solve_decomposed(LU, p, b)
            np.testing.assert_allclose(A @ x, b, rtol=1e-12)

    def test_matches_one_shot_solve(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        b = rng.random(5)
        x_oneshot = pylu.solve(A, b)
        LU, p = pylu.lup_packed(A)
        x_twostep = pylu.solve_decomposed(LU, p, b)
        np.testing.assert_allclose(x_oneshot, x_twostep, rtol=1e-14)


class TestBandedSolver:
    """Test find_bands() + solve_decomposed_banded() workflow."""

    def test_tridiagonal(self):
        n = 10
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = 2.0
            if i > 0:
                A[i, i-1] = -1.0
            if i < n - 1:
                A[i, i+1] = -1.0

        b = np.ones(n)
        LU, p = pylu.lup_packed(A)
        mincols, maxcols = pylu.find_bands(LU, 1e-15)
        x = pylu.solve_decomposed_banded(LU, p, mincols, maxcols, b)
        np.testing.assert_allclose(A @ x, b, rtol=1e-12)

    def test_banded_matches_non_banded(self):
        """Banded and non-banded solvers should give the same result."""
        rng = np.random.default_rng(42)
        n = 8
        A = rng.random((n, n))
        b = rng.random(n)

        LU, p = pylu.lup_packed(A)
        x_full = pylu.solve_decomposed(LU, p, b)

        mincols, maxcols = pylu.find_bands(LU, 1e-15)
        x_banded = pylu.solve_decomposed_banded(LU, p, mincols, maxcols, b)

        np.testing.assert_allclose(x_full, x_banded, rtol=1e-14)

    def test_diagonal_matrix(self):
        n = 5
        A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 6.0, 12.0, 20.0, 30.0])

        LU, p = pylu.lup_packed(A)
        mincols, maxcols = pylu.find_bands(LU, 1e-15)
        x = pylu.solve_decomposed_banded(LU, p, mincols, maxcols, b)
        np.testing.assert_allclose(x, [2.0, 3.0, 4.0, 5.0, 6.0], rtol=1e-14)

    def test_many_rhs_banded(self):
        """Factorize + find bands once, solve many RHS — the pydgq use case."""
        n = 6
        # Tridiagonal
        A = np.diag([2.0]*n) + np.diag([-1.0]*(n-1), k=1) + np.diag([-1.0]*(n-1), k=-1)

        LU, p = pylu.lup_packed(A)
        mincols, maxcols = pylu.find_bands(LU, 1e-15)

        rng = np.random.default_rng(42)
        for _ in range(20):
            b = rng.random(n)
            x = pylu.solve_decomposed_banded(LU, p, mincols, maxcols, b)
            np.testing.assert_allclose(A @ x, b, rtol=1e-12)

    def test_1x1_system(self):
        A = np.array([[3.0]])
        b = np.array([9.0])
        LU, p = pylu.lup_packed(A)
        mincols, maxcols = pylu.find_bands(LU, 1e-15)
        x = pylu.solve_decomposed_banded(LU, p, mincols, maxcols, b)
        np.testing.assert_allclose(x, [3.0], rtol=1e-14)


class TestAnalytical:
    """Analytical test case from the original test suite."""

    def test_known_system(self):
        A = np.array([[4, 1, 0, 0, 2],
                       [3, 3, -1, -1, 0],
                       [0, 0, 1, 1, 1],
                       [-2, 1, -2, 3, 7],
                       [0, 0, 2, 2, 1]], dtype=np.float64)
        b = np.array([19, 6, 8, 31, 11], dtype=np.float64)

        x = pylu.solve(A, b)
        np.testing.assert_allclose(A @ x, b, rtol=1e-12)

        # Also verify via factorize-then-solve path
        LU, p = pylu.lup_packed(A)
        x2 = pylu.solve_decomposed(LU, p, b)
        np.testing.assert_allclose(x, x2, rtol=1e-14)
