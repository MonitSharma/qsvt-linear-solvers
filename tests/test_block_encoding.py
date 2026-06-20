import numpy as np
import pytest

from primitives.block_encoding import (
    block_encoding_error,
    dilation_block_encoding,
    extract_block,
    is_unitary,
    lcu_block_encoding,
    normalize_matrix,
)

PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@pytest.mark.parametrize("n", [2, 3, 4])
def test_dilation_hermitian(n):
    rng = np.random.default_rng(n)
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M + M.conj().T
    A_n, alpha = normalize_matrix(A)
    U = dilation_block_encoding(A_n)
    assert is_unitary(U)
    assert block_encoding_error(U, A_n) < 1e-9
    # alpha * (encoded block) recovers A
    assert np.linalg.norm(alpha * extract_block(U, n) - A) < 1e-8


@pytest.mark.parametrize("n", [2, 4])
def test_dilation_nonhermitian(n):
    rng = np.random.default_rng(n + 10)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A_n, _ = normalize_matrix(A)
    U = dilation_block_encoding(A_n)
    assert is_unitary(U)
    assert block_encoding_error(U, A_n) < 1e-9


def test_dilation_rejects_non_contraction():
    with pytest.raises(ValueError):
        dilation_block_encoding(np.array([[2.0]]))


def test_lcu_block_encoding():
    coeffs = [0.5, 0.3, 0.2]
    unitaries = [PAULIS["X"], PAULIS["Y"], PAULIS["Z"]]
    W, alpha = lcu_block_encoding(coeffs, unitaries)
    assert alpha == pytest.approx(1.0)
    assert is_unitary(W)
    A = sum(c * U for c, U in zip(coeffs, unitaries))
    assert block_encoding_error(W, A / alpha) < 1e-9


def test_lcu_complex_coeffs():
    coeffs = [0.4 + 0.1j, -0.2j]
    unitaries = [PAULIS["X"], PAULIS["Z"]]
    W, alpha = lcu_block_encoding(coeffs, unitaries)
    A = sum(c * U for c, U in zip(coeffs, unitaries))
    assert is_unitary(W)
    assert block_encoding_error(W, A / alpha) < 1e-9
