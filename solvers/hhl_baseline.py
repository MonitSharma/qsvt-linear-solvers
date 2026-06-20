"""HHL baseline (QPE-based linear solver).

The Harrow-Hassidim-Lloyd algorithm solves ``A x = b`` by

1. quantum phase estimation (QPE) of ``e^{iAt}`` to write each eigenvalue
   ``lambda_j`` into a clock register,
2. an ancilla rotation ``R_y(2 arcsin(C / lambda_j))`` that loads ``1/lambda_j``,
3. uncomputing QPE and post-selecting the ancilla on ``|1>``.

We simulate it at the statevector level in the eigenbasis of ``A``, building the
*exact* QPE amplitude kernel (the Dirichlet kernel) for an ``m``-qubit clock.
This is circuit-faithful -- it reproduces HHL's phase-estimation error -- while
staying compact.  Its purpose is an honest comparison against the QSVT solver:

* QSVT error is controlled by polynomial degree (smooth, ``log(1/eps)``).
* HHL error is controlled by clock qubits ``m`` (phase-estimation resolution),
  and the success probability scales like ``O(1/kappa^2)`` without amplitude
  amplification.

This baseline assumes ``A`` is Hermitian positive-definite (the usual HHL demo
setting, e.g. the Poisson stiffness matrix).  Signed/­complex spectra need the
two's-complement QPE variant, which we deliberately omit to keep the baseline
readable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["HHLResult", "hhl_solve"]


@dataclass
class HHLResult:
    x: np.ndarray
    clock_qubits: int
    success_probability: float
    residual: float
    constant_C: float


def _qpe_amplitudes(theta: float, m: int) -> np.ndarray:
    """Exact QPE clock amplitudes for a true phase ``theta`` in [0, 1).

    Returns the length-``2^m`` vector ``alpha_y = (1/2^m) sum_x e^{2pi i x
    (theta - y/2^m)}`` -- the amplitude of measuring integer ``y`` in the clock.
    """
    M = 2**m
    y = np.arange(M)
    delta = theta - y / M
    # Geometric (Dirichlet) sum; handle delta ~ integer exactly.
    out = np.empty(M, dtype=complex)
    near = np.isclose(delta, np.round(delta))
    out[near] = 1.0
    d = delta[~near]
    out[~near] = (1 - np.exp(2j * np.pi * M * d)) / (M * (1 - np.exp(2j * np.pi * d)))
    return out


def hhl_solve(
    A: np.ndarray,
    b: np.ndarray,
    clock_qubits: int = 8,
    constant_C: float | None = None,
) -> HHLResult:
    """Solve ``A x = b`` with a faithful HHL statevector simulation.

    Parameters
    ----------
    A : (n, n) Hermitian positive-definite array.
    b : (n,) right-hand side.
    clock_qubits : int
        Size ``m`` of the QPE clock register (eigenvalue resolution ``2^-m``).
    constant_C : float, optional
        Rotation constant; defaults to a safe fraction of the smallest
        eigenvalue so all rotations are well defined.
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex).reshape(-1)
    if not np.allclose(A, A.conj().T, atol=1e-9):
        raise ValueError("HHL baseline requires a Hermitian matrix")

    evals, evecs = np.linalg.eigh(A)
    if np.any(evals <= 1e-12):
        raise ValueError("HHL baseline requires a positive-definite matrix")

    # Time scaling so that lambda * t / (2 pi) in (0, 1): keep largest phase < 1.
    lam_max = float(evals[-1])
    t = 2.0 * np.pi * (1.0 - 2.0 ** (-clock_qubits)) / lam_max
    thetas = evals * t / (2.0 * np.pi)  # true phases in (0, 1)

    if constant_C is None:
        constant_C = 0.5 * float(evals[0])  # < smallest eigenvalue

    M = 2 ** clock_qubits
    grid = np.arange(M)
    # Estimated eigenvalue for each clock outcome y.
    lam_est = np.where(grid > 0, (grid / M) * (2 * np.pi / t), np.inf)
    rot = np.where(np.isfinite(lam_est), constant_C / lam_est, 0.0)
    rot = np.clip(rot, -1.0, 1.0)  # |C/lambda| <= 1 for a valid rotation

    beta = evecs.conj().T @ b  # b in eigenbasis

    # Post-selected (ancilla=|1>) amplitude in eigenbasis:
    #   gamma_j = beta_j * sum_y |alpha_{j,y}|^2 ... but rotation acts per-y, so
    #   accumulate coherently over the clock then sum |.|^2 of the |1> branch.
    gamma = np.zeros(len(evals), dtype=complex)
    succ = 0.0
    for j in range(len(evals)):
        alpha = _qpe_amplitudes(float(thetas[j]), clock_qubits)
        # Amplitude into ancilla |1> for this eigenvector, summed over clock y
        # after uncomputing QPE (clock returns to |0>): coherent sum of alpha_y
        # weighted by the rotation sine.
        amp1 = np.sum(np.abs(alpha) ** 2 * rot)  # effective 1/lambda factor
        gamma[j] = beta[j] * amp1
    succ = float(np.sum(np.abs(gamma) ** 2))

    # Reconstruct x. The quantum output is a normalised state (direction only);
    # the overall scale is recovered classically by least-squares fitting the
    # single scalar s that minimises ||A (s x_dir) - b|| -- this uses only A, b
    # and the measured direction, never the true solution.
    x_dir = evecs @ gamma
    Ax = A @ x_dir
    denom = float(np.real(Ax.conj() @ Ax))
    if denom > 0:
        s = (Ax.conj() @ b) / denom
        x = s * x_dir
    else:
        x = x_dir
    residual = float(
        np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)
    )
    return HHLResult(
        x=x,
        clock_qubits=clock_qubits,
        success_probability=succ,
        residual=residual,
        constant_C=float(constant_C),
    )
