"""Block-encodings of matrices.

A matrix ``A`` (with spectral norm ``||A|| <= 1``) is *block-encoded* by a
unitary ``U`` acting on ``a`` ancilla qubits plus the system register when

    A = (<0|_a (x) I) U (|0>_a (x) I),

i.e. ``A`` is the top-left block of ``U`` selected by the ancilla being in
``|0...0>``.  This module provides two constructions:

* :func:`dilation_block_encoding` -- the textbook 1-ancilla unitary dilation of
  a contraction.  Exact, dense, and the workhorse used by the tests and the
  QSVT solver's verification path.
* :func:`lcu_block_encoding` -- a linear-combination-of-unitaries (LCU)
  block-encoding ``A = sum_i c_i U_i`` built from PREPARE/SELECT.  This is the
  construction that actually scales to sparse/structured operators on hardware.

Everything here works at the dense-matrix level so the maths can be checked
exactly against ``numpy``.  The hardware module turns the same objects into
``pytket`` circuits.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

__all__ = [
    "normalize_matrix",
    "dilation_block_encoding",
    "lcu_block_encoding",
    "extract_block",
    "block_encoding_error",
    "is_unitary",
]


def is_unitary(U: np.ndarray, atol: float = 1e-9) -> bool:
    """Return ``True`` if ``U`` is (numerically) unitary."""
    U = np.asarray(U)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        return False
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=atol)


def normalize_matrix(A: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale ``A`` to a contraction.

    Returns ``(A / alpha, alpha)`` where ``alpha = ||A||_2`` (the largest
    singular value).  A block-encoding of ``A / alpha`` together with ``alpha``
    is exactly a ``(alpha, .)``-block-encoding of ``A``.
    """
    A = np.asarray(A, dtype=complex)
    alpha = float(np.linalg.norm(A, ord=2))
    if alpha == 0.0:
        return A.copy(), 1.0
    return A / alpha, alpha


def _principal_sqrt_psd(M: np.ndarray) -> np.ndarray:
    """Hermitian PSD square root, numerically clean for ``I - A A^dag``."""
    M = (M + M.conj().T) / 2.0
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)  # guard tiny negative eigenvalues from round-off
    return (V * np.sqrt(w)) @ V.conj().T


def _defect_sqrts(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consistent left/right defect square roots for a contraction ``A``.

    If ``A = U S V^dag``, then

        sqrt(I - A A^dag) = U sqrt(I - S^2) U^dag
        sqrt(I - A^dag A) = V sqrt(I - S^2) V^dag.

    Building both factors from the same SVD preserves the identity
    ``A sqrt(I-A^dag A) = sqrt(I-AA^dag) A`` much more reliably than taking two
    independent eigendecompositions, especially after normalisation puts the
    largest singular value exactly at 1.
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    defect = np.sqrt(np.clip(1.0 - s * s, 0.0, None))
    V = Vh.conj().T
    left = (U * defect) @ U.conj().T
    right = (V * defect) @ V.conj().T
    return (left + left.conj().T) / 2.0, (right + right.conj().T) / 2.0


def dilation_block_encoding(A: np.ndarray, check: bool = True) -> np.ndarray:
    """One-ancilla unitary dilation of a contraction ``A`` (``||A|| <= 1``).

    Builds

        U = [[ A,                  sqrt(I - A A^dag) ],
             [ sqrt(I - A^dag A), -A^dag            ]]

    whose top-left block is ``A``.  ``U`` is unitary for any contraction; for
    Hermitian ``A`` it is additionally Hermitian (a reflection).

    Parameters
    ----------
    A : (n, n) array
        Must satisfy ``||A||_2 <= 1`` (use :func:`normalize_matrix` first).
    check : bool
        Verify the result is unitary and re-encodes ``A``.
    """
    A = np.asarray(A, dtype=complex)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("dilation_block_encoding requires a square matrix")
    smax = np.linalg.norm(A, ord=2)
    if smax > 1 + 1e-9:
        raise ValueError(
            f"||A|| = {smax:.6f} > 1; normalize A first (see normalize_matrix)"
        )

    top_right, bot_left = _defect_sqrts(A)
    U = np.block([[A, top_right], [bot_left, -A.conj().T]])

    if check:
        if not is_unitary(U):
            raise RuntimeError("dilation did not produce a unitary")
        err = block_encoding_error(U, A)
        if err > 1e-7:
            raise RuntimeError(f"dilation block mismatch: {err:.2e}")
    return U


def lcu_block_encoding(
    coeffs: np.ndarray, unitaries: list[np.ndarray], check: bool = True
) -> tuple[np.ndarray, float]:
    """Block-encode ``A = sum_i c_i U_i`` via PREPARE/SELECT (LCU).

    With ``alpha = sum_i |c_i|`` the returned unitary ``W = (PREP^dag (x) I)
    . SELECT . (PREP (x) I)`` is a ``(alpha, ceil(log2 L), 0)``-block-encoding
    of ``A``: its top-left block equals ``A / alpha``.

    Parameters
    ----------
    coeffs : (L,) array
        LCU coefficients ``c_i`` (may be complex).
    unitaries : list of (n, n) arrays
        The unitaries ``U_i``; must all be unitary and the same size.
    check : bool
        Verify unitarity of inputs/output and the encoded block.

    Returns
    -------
    (W, alpha) : the block-encoding unitary and the subnormalization factor.
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    L = len(coeffs)
    if L == 0 or len(unitaries) != L:
        raise ValueError("coeffs and unitaries must be non-empty and same length")
    n = unitaries[0].shape[0]
    if check:
        for i, Ui in enumerate(unitaries):
            if not is_unitary(Ui):
                raise ValueError(f"unitaries[{i}] is not unitary")

    alpha = float(np.sum(np.abs(coeffs)))
    if alpha == 0.0:
        raise ValueError("coefficients sum to zero")

    # Number of ancilla qubits / padded selector dimension.
    m = max(1, int(np.ceil(np.log2(L))))
    dim_anc = 2**m

    # PREPARE pair.  The encoded block is
    #   <0| PREP_L^dag . SELECT . PREP_R |0> = sum_i conj(L_i) R_i U_i,
    # so to recover sum_i c_i U_i we put the phases on the RIGHT prepare only and
    # keep the LEFT prepare real.  (Equal L = R would cancel phases and yield
    # sum_i |c_i| U_i instead.)
    amp = np.sqrt(np.abs(coeffs) / alpha)
    psi_right = np.zeros(dim_anc, dtype=complex)
    psi_right[:L] = amp * np.exp(1j * np.angle(coeffs))
    psi_left = np.zeros(dim_anc, dtype=complex)
    psi_left[:L] = amp
    prep_r = _complete_unitary(psi_right)
    prep_l = _complete_unitary(psi_left)

    # SELECT = sum_i |i><i| (x) U_i  (unused selector states act as identity).
    select = np.zeros((dim_anc * n, dim_anc * n), dtype=complex)
    for i in range(dim_anc):
        proj = np.zeros((dim_anc, dim_anc), dtype=complex)
        proj[i, i] = 1.0
        Ui = unitaries[i] if i < L else np.eye(n)
        select += np.kron(proj, Ui)

    W = np.kron(prep_l, np.eye(n)).conj().T @ select @ np.kron(prep_r, np.eye(n))

    if check:
        if not is_unitary(W):
            raise RuntimeError("LCU did not produce a unitary")
        A = sum(c * U for c, U in zip(coeffs, unitaries))
        err = block_encoding_error(W, A / alpha)
        if err > 1e-7:
            raise RuntimeError(f"LCU block mismatch: {err:.2e}")
    return W, alpha


def _complete_unitary(first_col: np.ndarray) -> np.ndarray:
    """Return a unitary whose first column is the (unit-norm) ``first_col``."""
    d = first_col.shape[0]
    M = np.eye(d, dtype=complex)
    M[:, 0] = first_col
    Q, R = np.linalg.qr(M)
    # Fix phases so column 0 matches first_col exactly.
    phase = first_col @ Q[:, 0].conj()
    Q[:, 0] *= phase / abs(phase) if abs(phase) > 0 else 1.0
    if not np.allclose(Q[:, 0], first_col, atol=1e-9):
        # QR can flip the sign; align explicitly.
        Q[:, 0] = first_col
        Q, _ = np.linalg.qr(Q)
        Q[:, 0] = first_col
    return Q


def extract_block(U: np.ndarray, n: int) -> np.ndarray:
    """Top-left ``n x n`` block of ``U`` (the encoded operator)."""
    return np.asarray(U)[:n, :n]


def block_encoding_error(U: np.ndarray, A: np.ndarray) -> float:
    """Spectral-norm distance between the encoded block of ``U`` and ``A``."""
    A = np.asarray(A)
    block = extract_block(U, A.shape[0])
    return float(np.linalg.norm(block - A, ord=2))
