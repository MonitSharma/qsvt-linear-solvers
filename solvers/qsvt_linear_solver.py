"""QSVT linear-equation solver.

Solve ``A x = b`` by applying a polynomial approximation of ``1/x`` to a
block-encoding of ``A`` via QSVT:

1. Normalise ``A -> A_n = A / alpha`` so ``||A_n|| <= 1`` (``alpha = ||A||_2``).
2. Estimate the condition number ``kappa`` and build an odd polynomial
   ``P(x) ~ scale / x`` accurate on the spectral band ``[1/kappa, 1]``.
3. Find QSP phase factors for ``P`` and apply QSVT to the ``W_x`` block-encoding
   of ``A_n``, yielding (the encoded block) ``P(A_n) ~ scale * A_n^{-1}``.
4. Act on ``|b>`` and rescale: ``x = A^{-1} b ~ P(A_n) b / (scale * alpha)``.

The class operates at the statevector / dense-matrix level: it is a faithful
*simulation* of the fault-tolerant algorithm (block-encode -> QSVT -> amplitude
amplification -> read out), exact up to the polynomial approximation error.  The
same ``A_n``, phases and block-encoding drive the pytket circuit in
``hardware/``.

Non-Hermitian ``A`` is handled by the standard Hermitian dilation
``H = [[0, A], [A^dag, 0]]`` with right-hand side ``[b; 0]``; the solution
appears in the lower block.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from primitives.amplitude_amplification import optimal_rounds
from primitives.qsp_qsvt import (
    PolynomialApproximation,
    approximate_inverse,
    find_phases,
    qsvt_matrix_function,
    wx_block_encoding,
)

__all__ = ["LinearSolveResult", "QSVTLinearSolver", "solve"]


@dataclass
class LinearSolveResult:
    """Outcome of a QSVT linear solve."""

    x: np.ndarray  # recovered solution vector (full scale)
    kappa: float
    degree: int
    scale: float
    alpha: float
    success_probability: float  # prob. of the ancilla=|0> read-out
    amplification_rounds: int  # Grover rounds to make that O(1)
    residual: float  # ||A x - b|| / ||b||
    polynomial: PolynomialApproximation = field(repr=False)


class QSVTLinearSolver:
    """Configurable QSVT-based solver for ``A x = b``."""

    def __init__(
        self,
        epsilon: float = 0.01,
        kappa_safety: float = 1.2,
        bound: float = 0.9,
        phase_method: str = "auto",
    ):
        self.epsilon = epsilon
        self.kappa_safety = kappa_safety
        self.bound = bound
        self.phase_method = phase_method

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _is_hermitian(A: np.ndarray) -> bool:
        return np.allclose(A, A.conj().T, atol=1e-9)

    @staticmethod
    def _hermitian_dilation(A: np.ndarray, b: np.ndarray):
        n = A.shape[0]
        H = np.zeros((2 * n, 2 * n), dtype=complex)
        H[:n, n:] = A
        H[n:, :n] = A.conj().T
        rhs = np.concatenate([b, np.zeros(n, dtype=complex)])
        return H, rhs

    # -- main ------------------------------------------------------------- #
    def solve(self, A: np.ndarray, b: np.ndarray) -> LinearSolveResult:
        A = np.asarray(A, dtype=complex)
        b = np.asarray(b, dtype=complex).reshape(-1)
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        if A.shape[0] != b.shape[0]:
            raise ValueError("dimension mismatch between A and b")

        if self._is_hermitian(A):
            return self._solve_hermitian(A, b)

        # General A: solve the Hermitian dilation, extract the lower block.
        H, rhs = self._hermitian_dilation(A, b)
        sub = self._solve_hermitian(H, rhs)
        n = A.shape[0]
        x = sub.x[n:]
        residual = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
        return LinearSolveResult(
            x=x,
            kappa=sub.kappa,
            degree=sub.degree,
            scale=sub.scale,
            alpha=sub.alpha,
            success_probability=sub.success_probability,
            amplification_rounds=sub.amplification_rounds,
            residual=residual,
            polynomial=sub.polynomial,
        )

    def _solve_hermitian(self, A: np.ndarray, b: np.ndarray) -> LinearSolveResult:
        # 1. normalise so ||A_n|| <= 1
        sv = np.linalg.svd(A, compute_uv=False)
        alpha = float(sv[0])
        smin = float(sv[-1])
        if smin <= 1e-12:
            raise ValueError("A is singular (or numerically so); cannot invert")
        A_n = A / alpha
        kappa = self.kappa_safety * alpha / smin

        # 2. polynomial approximation of 1/x on [1/kappa, 1]
        poly = approximate_inverse(kappa, epsilon=self.epsilon, bound=self.bound)

        # 3. phases + QSVT  ->  P(A_n) ~ scale * A_n^{-1}
        phis = find_phases(poly.cheb_coeffs, poly.parity, method=self.phase_method)
        _ = wx_block_encoding(A_n)  # validates A_n is a usable block-encoding input
        P_An = qsvt_matrix_function(A_n, phis)

        # 4. apply to |b> and undo the subnormalisations
        bnorm = float(np.linalg.norm(b))
        b_hat = b / bnorm
        encoded = P_An @ b_hat  # ~ scale * A_n^{-1} b_hat (lives in ancilla=|0>)
        success_probability = float(np.linalg.norm(encoded) ** 2)

        # x = A^{-1} b = (1/alpha) A_n^{-1} b ~ P(A_n) b / (scale * alpha)
        x = (P_An @ b) / (poly.scale * alpha)

        residual = float(np.linalg.norm(A @ x - b) / max(bnorm, 1e-300))
        return LinearSolveResult(
            x=x,
            kappa=kappa,
            degree=poly.degree,
            scale=poly.scale,
            alpha=alpha,
            success_probability=success_probability,
            amplification_rounds=optimal_rounds(success_probability),
            residual=residual,
            polynomial=poly,
        )


def solve(A: np.ndarray, b: np.ndarray, **kwargs) -> LinearSolveResult:
    """Convenience wrapper: ``solve(A, b)`` with default solver settings."""
    return QSVTLinearSolver(**kwargs).solve(A, b)
