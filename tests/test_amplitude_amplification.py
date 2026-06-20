import numpy as np

from primitives.amplitude_amplification import (
    amplitude_amplification,
    fixed_point_amplification,
    optimal_rounds,
    success_probability,
)


def _prep_and_projector(N, good_index, p0, seed=0):
    """A state-prep unitary whose |psi> has success amplitude sqrt(p0) on a
    one-dimensional good subspace."""
    rng = np.random.default_rng(seed)
    # Build |psi> with the desired overlap, then a unitary with it as column 0.
    psi = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    psi[good_index] = 0
    psi = psi / np.linalg.norm(psi) * np.sqrt(1 - p0)
    psi[good_index] = np.sqrt(p0)
    M = np.eye(N, dtype=complex)
    M[:, 0] = psi
    U, _ = np.linalg.qr(M)
    U[:, 0] = psi  # ensure column 0 is exactly |psi>
    U, _ = np.linalg.qr(U)
    U[:, 0] = psi
    P = np.zeros((N, N), dtype=complex)
    P[good_index, good_index] = 1.0
    return U, P


def test_amplification_boosts_small_amplitude():
    U, P = _prep_and_projector(8, good_index=3, p0=0.05, seed=1)
    psi0 = U @ np.eye(8)[:, 0]
    assert success_probability(psi0, P) < 0.1
    _, p = amplitude_amplification(U, P)
    assert p > 0.8


def test_optimal_rounds_matches_theory():
    # p0 = sin^2(theta); optimal k ~ pi/(4 theta) - 1/2
    p0 = 0.04
    theta = np.arcsin(np.sqrt(p0))
    expected = round((np.pi / (2 * theta) - 1) / 2)
    assert optimal_rounds(p0) == expected


def test_fixed_point_is_monotone_and_high():
    U, P = _prep_and_projector(8, good_index=2, p0=0.1, seed=4)
    psi0 = U @ np.eye(8)[:, 0]
    p_start = success_probability(psi0, P)
    probs = []
    for it in range(5):
        _, p = fixed_point_amplification(U, P, iterations=it)
        probs.append(p)
    assert probs[0] >= p_start - 1e-9
    # pi/3 recursion drives failure amplitude as eps -> eps^3 each level.
    assert probs[-1] > 0.99
    # monotone non-decreasing (no overshoot)
    assert all(probs[i + 1] >= probs[i] - 1e-9 for i in range(len(probs) - 1))
