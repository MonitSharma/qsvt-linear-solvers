import numpy as np
import pytest
from numpy.polynomial import chebyshev as C

from primitives.block_encoding import is_unitary
from primitives.qsp_qsvt import (
    approximate_inverse,
    find_phases,
    qsp_response,
    qsvt_matrix_function,
    wx_block_encoding,
)


def _random_hermitian_in_band(n, kappa, seed):
    """Hermitian matrix with eigenvalues in magnitude band [1/kappa, 1]."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    mags = rng.uniform(1.0 / kappa, 1.0, n)
    signs = rng.choice([-1.0, 1.0], n)
    lam = mags * signs
    A = (Q * lam) @ Q.conj().T
    return (A + A.conj().T) / 2.0


def test_approximate_inverse_is_bounded_and_accurate():
    kappa = 5
    poly = approximate_inverse(kappa, epsilon=0.05)
    assert poly.parity == 1
    assert poly.max_abs_on_interval() <= 1.0
    xs = np.linspace(1 / kappa, 1.0, 50)
    # P(x) ~ scale / x  =>  P(x) * x ~ scale (constant)
    prod = poly(xs) * xs
    assert np.allclose(prod, poly.scale, atol=5e-2)


def test_find_phases_reconstructs_polynomial():
    poly = approximate_inverse(4, epsilon=0.1)
    phis = find_phases(poly.cheb_coeffs, poly.parity)
    assert len(phis) == len(poly.cheb_coeffs)
    xs = np.linspace(1 / 4, 1.0, 40)
    realized = np.array([qsp_response(phis, x).imag for x in xs])
    assert np.max(np.abs(realized - poly(xs))) < 1e-6


def test_find_phases_optimize_fallback_low_degree():
    # A small odd polynomial QSP can represent exactly: 0.5 * T_3.
    coeffs = np.zeros(4)
    coeffs[3] = 0.5
    phis = find_phases(coeffs, parity=1, method="optimize")
    xs = np.linspace(-0.9, 0.9, 25)
    realized = np.array([qsp_response(phis, x).imag for x in xs])
    assert np.max(np.abs(realized - C.chebval(xs, coeffs))) < 1e-4


def test_wx_block_encoding_unitary_and_block():
    A = _random_hermitian_in_band(4, kappa=4, seed=1)
    U = wx_block_encoding(A)
    assert is_unitary(U)
    n = A.shape[0]
    assert np.linalg.norm(U[:n, :n] - A) < 1e-9


def test_qsvt_realizes_matrix_inverse():
    kappa = 4
    A = _random_hermitian_in_band(4, kappa=kappa, seed=2)
    poly = approximate_inverse(kappa, epsilon=0.05)
    phis = find_phases(poly.cheb_coeffs, poly.parity)
    P_A = qsvt_matrix_function(A, phis)

    # exact target: V P(lambda) V^dag
    w, V = np.linalg.eigh(A)
    target = (V * poly(w)) @ V.conj().T
    assert np.linalg.norm(P_A - target, 2) < 1e-9

    # and P(A) ~ scale * A^{-1}
    assert np.linalg.norm(P_A - poly.scale * np.linalg.inv(A), 2) < 0.1
