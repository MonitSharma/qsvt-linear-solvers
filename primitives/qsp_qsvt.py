"""Quantum signal processing (QSP) and quantum singular value transformation
(QSVT).

This module covers the three ingredients needed to apply a matrix function via
QSVT:

1. **Polynomial approximation** of a target function on the spectrum -- here the
   reciprocal ``1/x`` on the well-conditioned band ``[1/kappa, 1]`` (plus its
   reflection), which is what the linear solver needs.
2. **Phase-factor (angle) finding** -- given a real, definite-parity polynomial
   ``P`` with ``|P| <= 1``, find phases ``{phi_k}`` so that a QSP sequence
   realises ``P``.
3. **QSVT application** -- turn a block-encoding ``U_A`` of a Hermitian ``A``
   into a block-encoding of ``P(A)``.

Conventions
-----------
We use the ``W_x`` signal convention throughout::

    W(x) = [[x, i sqrt(1-x^2)], [i sqrt(1-x^2), x]]
    S(phi) = diag(e^{i phi}, e^{-i phi})            # = e^{i phi Z}
    U_Phi(x) = S(phi_0) prod_{k>=1} W(x) S(phi_k)

With *symmetric* phase factors the target polynomial is realised as the
**imaginary part** of the top-left element::

    Im <0| U_Phi(x) |0> = P(x).

Accordingly QSVT realises ``P(A)`` as the anti-symmetric combination
``(U_Phi - U_{-Phi}) / (2i)`` (an "imaginary-part" Hadamard test / LCU), whose
encoded block equals ``P(A)`` exactly.

Angle finding
-------------
Robust phase-factor evaluation at high degree is a genuinely hard numerical
problem (the reason the literature uses high-precision symmetric-QSP solvers).
We delegate it to :mod:`pyqsp`'s ``sym_qsp`` Newton solver (QSPPACK Alg. 3.3)
when available, and fall back to a small from-scratch least-squares optimiser
that is reliable only at low degree.  Everything else in this file -- the signal
convention, the QSVT operator sequence, the imaginary-part extraction and the
polynomial layer -- is implemented here directly and checked numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from numpy.polynomial import chebyshev as _C
from scipy.optimize import least_squares

__all__ = [
    "PolynomialApproximation",
    "approximate_inverse",
    "find_phases",
    "qsp_response",
    "wx_block_encoding",
    "qsvt_sequence",
    "qsvt_block_operator",
    "qsvt_matrix_function",
]


# --------------------------------------------------------------------------- #
# 1. Polynomial approximation
# --------------------------------------------------------------------------- #
@dataclass
class PolynomialApproximation:
    """An odd/even Chebyshev polynomial approximating a target function.

    Attributes
    ----------
    cheb_coeffs : np.ndarray
        Coefficients in the Chebyshev basis (``numpy.polynomial.chebyshev``).
    parity : int
        ``0`` for even, ``1`` for odd.
    scale : float
        Subnormalisation: the polynomial approximates ``scale * f(x)`` for the
        nominal function ``f``.  For ``1/x`` this means ``P(x) ~ scale / x``.
    kind : str
        Human-readable label of the approximated function.
    """

    cheb_coeffs: np.ndarray
    parity: int
    scale: float
    kind: str = "custom"

    def __call__(self, x):
        return _C.chebval(x, self.cheb_coeffs)

    @property
    def degree(self) -> int:
        return len(self.cheb_coeffs) - 1

    def max_abs_on_interval(self, lo: float = -1.0, hi: float = 1.0, n: int = 2000) -> float:
        xs = np.linspace(lo, hi, n)
        return float(np.max(np.abs(self(xs))))


def approximate_inverse(
    kappa: float, epsilon: float = 0.05, bound: float = 0.9
) -> PolynomialApproximation:
    """Odd polynomial approximating ``1/x`` on ``[1/kappa, 1] u [-1, -1/kappa]``.

    Uses :mod:`pyqsp`'s ``PolyOneOverX`` construction (Childs-Kothari-Somma style)
    which returns a bounded Chebyshev polynomial ``P(x) ~ scale / x``.

    Parameters
    ----------
    kappa : float
        Condition number; the approximation is accurate for ``|x| in [1/kappa, 1]``.
    epsilon : float
        Target approximation error; smaller ``epsilon`` -> higher degree.
    bound : float
        Maximum amplitude of the polynomial on ``[-1, 1]`` (must be < 1 for QSP).
    """
    try:
        from pyqsp.poly import PolyOneOverX
    except ImportError as exc:  # pragma: no cover - exercised only without pyqsp
        raise ImportError(
            "approximate_inverse needs pyqsp for the bounded 1/x polynomial; "
            "install it with `pip install pyqsp`."
        ) from exc

    coeffs, scale = PolyOneOverX().generate(
        kappa=kappa, epsilon=epsilon, return_scale=True, chebyshev_basis=True
    )
    coeffs = np.asarray(coeffs, dtype=float).flatten()
    scale = float(np.asarray(scale).flatten()[0])
    return PolynomialApproximation(
        cheb_coeffs=coeffs, parity=1, scale=scale, kind="inverse(1/x)"
    )


# --------------------------------------------------------------------------- #
# 2. Phase-factor (angle) finding
# --------------------------------------------------------------------------- #
def _qsp_unitary_scalar(phis: np.ndarray, x: float) -> np.ndarray:
    s = np.sqrt(max(0.0, 1.0 - x * x))
    W = np.array([[x, 1j * s], [1j * s, x]], dtype=complex)
    U = np.array(
        [[np.exp(1j * phis[0]), 0.0], [0.0, np.exp(-1j * phis[0])]], dtype=complex
    )
    for phi in phis[1:]:
        S = np.array([[np.exp(1j * phi), 0.0], [0.0, np.exp(-1j * phi)]], dtype=complex)
        U = U @ W @ S
    return U


def qsp_response(phis: np.ndarray, x: float) -> complex:
    """Top-left element ``<0| U_Phi(x) |0>`` of the QSP sequence.

    With symmetric phases, ``Im(qsp_response) = P(x)``.
    """
    return complex(_qsp_unitary_scalar(np.asarray(phis, dtype=float), float(x))[0, 0])


def _find_phases_pyqsp(cheb_coeffs: np.ndarray, parity: int) -> np.ndarray:
    from pyqsp.sym_qsp_opt import newton_solver

    reduced_coefs = np.asarray(cheb_coeffs, dtype=float)[parity::2]
    _, _, _, protocol = newton_solver(reduced_coefs, parity, crit=1e-12)
    return np.asarray(protocol.full_phases, dtype=float).flatten()


def _find_phases_optimize(
    cheb_coeffs: np.ndarray, parity: int, npts: int | None = None
) -> np.ndarray:
    """From-scratch symmetric-QSP least squares (reliable only at low degree)."""
    d = len(cheb_coeffs) - 1
    npts = npts or (4 * d + 20)
    nodes = np.cos(np.pi * (np.arange(npts) + 0.5) / npts)
    nodes = nodes[nodes > 0]
    target = _C.chebval(nodes, cheb_coeffs)

    n_free = (d + 1 + 1) // 2

    def expand(free: np.ndarray) -> np.ndarray:
        return np.array([free[min(k, d - k)] for k in range(d + 1)])

    def residual(free: np.ndarray) -> np.ndarray:
        phis = expand(free)
        return np.array(
            [qsp_response(phis, x).imag - t for x, t in zip(nodes, target)]
        )

    sol = least_squares(residual, np.zeros(n_free), method="lm", max_nfev=50000)
    phis = expand(sol.x)
    return phis


def find_phases(
    cheb_coeffs: np.ndarray, parity: int, method: str = "auto"
) -> np.ndarray:
    """Find symmetric QSP phase factors realising the Chebyshev polynomial.

    Parameters
    ----------
    cheb_coeffs : array
        Chebyshev coefficients of a real, definite-parity polynomial ``P`` with
        ``|P(x)| <= 1`` on ``[-1, 1]``.
    parity : int
        ``0`` (even) or ``1`` (odd); must match ``cheb_coeffs``.
    method : {"auto", "pyqsp", "optimize"}
        ``"auto"`` uses pyqsp's robust solver if installed, else the local
        optimiser.

    Returns
    -------
    np.ndarray
        Full phase list ``[phi_0, ..., phi_d]`` (length ``len(cheb_coeffs)``)
        such that ``Im qsp_response(phis, x) = P(x)``.
    """
    cheb_coeffs = np.asarray(cheb_coeffs, dtype=float)
    if method == "optimize":
        return _find_phases_optimize(cheb_coeffs, parity)
    if method == "pyqsp":
        return _find_phases_pyqsp(cheb_coeffs, parity)
    if method == "auto":
        try:
            return _find_phases_pyqsp(cheb_coeffs, parity)
        except ImportError:
            return _find_phases_optimize(cheb_coeffs, parity)
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------------- #
# 3. QSVT application
# --------------------------------------------------------------------------- #
def wx_block_encoding(A: np.ndarray) -> np.ndarray:
    """``W_x``-convention block-encoding of a Hermitian contraction ``A``.

    Returns the unitary

        U_A = [[A,            i sqrt(I - A^2)],
               [i sqrt(I-A^2), A             ]]

    which acts on each eigenvector of ``A`` (eigenvalue ``lambda``) as the
    scalar signal ``W(lambda)``.  Requires ``A`` Hermitian with ``||A|| <= 1``.
    """
    A = np.asarray(A, dtype=complex)
    n = A.shape[0]
    if not np.allclose(A, A.conj().T, atol=1e-9):
        raise ValueError("wx_block_encoding expects a Hermitian matrix")
    if np.linalg.norm(A, ord=2) > 1 + 1e-9:
        raise ValueError("||A|| > 1; normalize A first")
    eye = np.eye(n)
    root = sla.sqrtm(eye - A @ A)
    root = (root + root.conj().T) / 2.0  # clean numerical Hermiticity
    U = np.block([[A, 1j * root], [1j * root, A]])
    return U


def _ancilla_phase(phi: float, n: int) -> np.ndarray:
    eye = np.eye(n)
    z = np.zeros((n, n))
    return np.block(
        [[np.exp(1j * phi) * eye, z], [z, np.exp(-1j * phi) * eye]]
    )


def qsvt_sequence(U_A: np.ndarray, phis: np.ndarray) -> np.ndarray:
    """Interleaved QSVT operator sequence ``S(phi_0) prod_k U_A S(phi_k)``.

    ``U_A`` must be a ``W_x``-convention block-encoding (see
    :func:`wx_block_encoding`).
    """
    U_A = np.asarray(U_A, dtype=complex)
    n = U_A.shape[0] // 2
    phis = np.asarray(phis, dtype=float)
    Q = _ancilla_phase(phis[0], n)
    for phi in phis[1:]:
        Q = Q @ U_A @ _ancilla_phase(phi, n)
    return Q


def qsvt_block_operator(U_A: np.ndarray, phis: np.ndarray) -> np.ndarray:
    """Encoded block ``P(A)`` realised by QSVT, via the imaginary-part LCU.

    Returns the ``n x n`` operator ``(Q_Phi - Q_{-Phi}) / (2i)`` restricted to
    the encoded block, which equals ``P(A)`` for the polynomial defined by
    ``phis``.
    """
    n = np.asarray(U_A).shape[0] // 2
    Qp = qsvt_sequence(U_A, phis)
    Qm = qsvt_sequence(U_A, -np.asarray(phis, dtype=float))
    return ((Qp - Qm) / (2j))[:n, :n]


def qsvt_matrix_function(A: np.ndarray, phis: np.ndarray) -> np.ndarray:
    """Convenience: build the ``W_x`` block-encoding of ``A`` and return ``P(A)``."""
    return qsvt_block_operator(wx_block_encoding(A), phis)
