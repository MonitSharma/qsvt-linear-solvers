"""Amplitude amplification.

After QSVT applies ``P(A) ~ scale * A^{-1}`` to ``|b>``, the (unnormalised)
solution lives in the ancilla-``|0>`` block and is reached only with probability
``p = || scale * A^{-1} |b> ||^2``.  Amplitude amplification boosts that
probability so the solution can be read out with O(1/sqrt(p)) rather than O(1/p)
repetitions.

This module implements the constructions at the statevector level so they can
be verified exactly:

* :func:`amplitude_amplification` -- standard (Grover/Brassard) amplification of
  a marked subspace defined by a projector.
* :func:`fixed_point_amplification` -- a robust, overshoot-free schedule (the
  pi/3 fixed-point sequence) for when the initial amplitude is only known to lie
  in a range.
* :func:`optimal_rounds` / :func:`success_probability` -- helpers for planning
  and checking.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "reflection",
    "grover_operator",
    "success_probability",
    "optimal_rounds",
    "amplitude_amplification",
    "fixed_point_amplification",
]


def reflection(projector: np.ndarray) -> np.ndarray:
    """Reflection ``I - 2 P`` about the complement of the subspace ``P``."""
    P = np.asarray(projector, dtype=complex)
    return np.eye(P.shape[0]) - 2.0 * P


def success_probability(state: np.ndarray, projector: np.ndarray) -> float:
    """Probability ``<psi| P |psi>`` of measuring inside the marked subspace."""
    psi = np.asarray(state, dtype=complex).reshape(-1)
    P = np.asarray(projector, dtype=complex)
    return float(np.real(psi.conj() @ (P @ psi)))


def grover_operator(
    state_prep: np.ndarray, target_projector: np.ndarray
) -> np.ndarray:
    """Grover/amplification iterate ``Q = -R_psi R_target``.

    Parameters
    ----------
    state_prep : (N, N) unitary
        The unitary ``U`` preparing the initial state ``|psi> = U|0>`` (so the
        ``|0>`` reflection is ``U (I - 2|0><0|) U^dag``).
    target_projector : (N, N)
        Projector ``P`` onto the marked ("good") subspace.
    """
    U = np.asarray(state_prep, dtype=complex)
    N = U.shape[0]
    zero = np.zeros((N, N), dtype=complex)
    zero[0, 0] = 1.0
    R_psi = U @ reflection(zero) @ U.conj().T  # reflection about |psi>
    R_target = reflection(target_projector)  # reflection about good subspace
    return -R_psi @ R_target


def optimal_rounds(p0: float) -> int:
    """Number of Grover rounds that maximises success from initial prob ``p0``."""
    p0 = min(max(p0, 1e-15), 1.0)
    theta = np.arcsin(np.sqrt(p0))
    # Amplitude after k rounds ~ sin((2k+1) theta); maximise -> (2k+1)theta ~ pi/2.
    k = (np.pi / (2 * theta) - 1) / 2
    return max(0, int(round(k)))


def amplitude_amplification(
    state_prep: np.ndarray,
    target_projector: np.ndarray,
    rounds: int | None = None,
) -> tuple[np.ndarray, float]:
    """Amplify the marked subspace and return ``(final_state, success_prob)``.

    If ``rounds`` is ``None`` it is chosen by :func:`optimal_rounds` from the
    initial success probability.
    """
    U = np.asarray(state_prep, dtype=complex)
    N = U.shape[0]
    e0 = np.zeros(N, dtype=complex)
    e0[0] = 1.0
    psi = U @ e0
    if rounds is None:
        rounds = optimal_rounds(success_probability(psi, target_projector))
    Q = grover_operator(U, target_projector)
    for _ in range(rounds):
        psi = Q @ psi
    return psi, success_probability(psi, target_projector)


def fixed_point_amplification(
    state_prep: np.ndarray,
    target_projector: np.ndarray,
    iterations: int = 3,
) -> tuple[np.ndarray, float]:
    """Grover's pi/3 fixed-point amplification.

    Monotonically drives the success probability toward 1 without the overshoot
    of fixed-angle amplification, at the cost of a recursively defined schedule.
    Useful when the initial amplitude is uncertain (as it is for an unknown RHS).
    """
    U = np.asarray(state_prep, dtype=complex)
    N = U.shape[0]
    zero = np.zeros((N, N), dtype=complex)
    zero[0, 0] = 1.0
    P = np.asarray(target_projector, dtype=complex)

    def phase_reflection(proj: np.ndarray, phase: float) -> np.ndarray:
        return np.eye(proj.shape[0]) - (1 - np.exp(1j * phase)) * proj

    e0 = np.zeros(N, dtype=complex)
    e0[0] = 1.0

    # Recursive pi/3 sequence U_{n+1} = U_n S_s(pi/3) U_n^dag S_t(pi/3) U_n.
    def apply(level: int, vec: np.ndarray) -> np.ndarray:
        if level == 0:
            return U @ vec
        St = phase_reflection(P, np.pi / 3)
        Ss = phase_reflection(zero, np.pi / 3)
        v = apply(level - 1, vec)
        v = St @ v
        v = apply_dagger(level - 1, v)
        v = Ss @ v
        v = apply(level - 1, v)
        return v

    def apply_dagger(level: int, vec: np.ndarray) -> np.ndarray:
        if level == 0:
            return U.conj().T @ vec
        St = phase_reflection(P, -np.pi / 3)
        Ss = phase_reflection(zero, -np.pi / 3)
        v = apply_dagger(level - 1, vec)
        v = Ss @ v
        v = apply(level - 1, v)
        v = St @ v
        v = apply_dagger(level - 1, v)
        return v

    psi = apply(iterations, e0)
    return psi, success_probability(psi, P)
