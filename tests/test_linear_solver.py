import numpy as np
import pytest

from solvers.hhl_baseline import hhl_solve
from solvers.qsvt_linear_solver import solve


def _spd(n, seed, shift=3.0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return M @ M.conj().T + shift * np.eye(n)


def test_qsvt_solves_hermitian_spd():
    A = _spd(4, seed=1)
    rng = np.random.default_rng(2)
    b = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    res = solve(A, b, epsilon=0.01)
    assert res.residual < 1e-3
    assert np.allclose(res.x, np.linalg.solve(A, b), atol=1e-3)
    assert 0.0 < res.success_probability <= 1.0


def test_qsvt_solves_indefinite_hermitian():
    # eigenvalues of both signs, well away from zero
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
    lam = np.array([-2.0, -1.0, 1.5, 3.0])
    A = (Q * lam) @ Q.conj().T
    A = (A + A.conj().T) / 2
    b = rng.standard_normal(4)
    res = solve(A, b, epsilon=0.01)
    assert res.residual < 1e-2


def test_qsvt_solves_nonhermitian():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)) + 3 * np.eye(4)
    b = rng.standard_normal(4)
    res = solve(A, b, epsilon=0.01)
    assert res.residual < 1e-2
    assert np.allclose(res.x, np.linalg.solve(A, b), atol=1e-2)


def test_qsvt_rejects_singular():
    A = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
    with pytest.raises(ValueError):
        solve(A, np.array([1.0, 0.0]))


def test_hhl_converges_with_clock_qubits():
    A = _spd(4, seed=11).real  # real SPD
    A = (A + A.T) / 2
    rng = np.random.default_rng(12)
    b = rng.standard_normal(4)
    r_coarse = hhl_solve(A, b, clock_qubits=5)
    r_fine = hhl_solve(A, b, clock_qubits=9)
    assert r_fine.residual < r_coarse.residual
    assert r_fine.residual < 1e-2


def test_qsvt_and_hhl_agree_on_solution():
    A = _spd(4, seed=21).real
    A = (A + A.T) / 2
    rng = np.random.default_rng(22)
    b = rng.standard_normal(4)
    x_qsvt = solve(A, b, epsilon=0.01).x
    x_hhl = hhl_solve(A, b, clock_qubits=9).x
    x_true = np.linalg.solve(A, b)
    assert np.linalg.norm(x_qsvt - x_true) / np.linalg.norm(x_true) < 1e-2
    assert np.linalg.norm(x_hhl - x_true) / np.linalg.norm(x_true) < 1e-2
