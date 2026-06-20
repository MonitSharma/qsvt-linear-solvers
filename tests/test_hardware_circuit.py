"""Hardware-circuit tests.

These exercise the pytket circuit construction only (no Nexus / no network).
Skipped automatically where pytket is not installed (e.g. the lightweight CI
job), so the core suite stays dependency-light.
"""

import numpy as np
import pytest

pytest.importorskip("pytket")

from primitives.qsp_qsvt import (  # noqa: E402
    approximate_inverse,
    find_phases,
    qsvt_sequence,
    wx_block_encoding,
)


def _setup():
    A = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=complex)
    b = np.array([1.0, 0.0], dtype=complex)
    poly = approximate_inverse(kappa=2.0, epsilon=0.3)
    phis = find_phases(poly.cheb_coeffs, poly.parity)
    return A, b, phis


def test_circuit_matches_matrix_qsvt():
    from hardware.quantinuum_runner import build_qsvt_circuit, simulate_statevector

    A, b, phis = _setup()
    circ = build_qsvt_circuit(A, b=b, phis=phis, measure=False)
    sv = simulate_statevector(circ)

    Q = qsvt_sequence(wx_block_encoding(A), phis)
    ref = Q @ np.kron([1, 0], b / np.linalg.norm(b))

    # equal up to a global phase
    phase = np.vdot(sv, ref)
    phase /= abs(phase)
    assert np.allclose(sv, ref * phase, atol=1e-9)


def test_circuit_rebases_to_helios_native_gates():
    from pytket.circuit import OpType
    from pytket.passes import AutoRebase, DecomposeBoxes, FullPeepholeOptimise

    from hardware.quantinuum_runner import build_qsvt_circuit

    A, b, phis = _setup()
    circ = build_qsvt_circuit(A, b=b, phis=phis, measure=True)
    DecomposeBoxes().apply(circ)
    FullPeepholeOptimise().apply(circ)
    native = {OpType.ZZPhase, OpType.PhasedX, OpType.Rz, OpType.Measure}
    AutoRebase(native).apply(circ)

    types = {cmd.op.type for cmd in circ.get_commands()}
    assert types.issubset(native)


def test_circuit_to_hugr_builds_valid_main():
    pytest.importorskip("tket")
    from tket._state import CompilationState

    from hardware.quantinuum_runner import build_qsvt_circuit, circuit_to_hugr

    A, b, phis = _setup()
    circ = build_qsvt_circuit(A, b=b, phis=phis, measure=True)
    pkg = circuit_to_hugr(circ)
    # Round-trip through the HUGR validator: a no-input main() entry point with
    # the tket extensions resolved is exactly what the Helios runtime requires.
    CompilationState.from_str(pkg.to_str()).validate()


def test_ibm_qiskit_circuit_matches_matrix_qsvt():
    pytest.importorskip("qiskit")

    from hardware.run_ibm import circuit_ideal

    A, b, phis = _setup()
    ideal = circuit_ideal(A, b, phis)

    Q = qsvt_sequence(wx_block_encoding(A), phis)
    ref = Q @ np.kron([1, 0], b / np.linalg.norm(b))
    probs = np.abs(ref) ** 2

    # Qiskit count keys are displayed as system,ancilla.  The matrix model above
    # is indexed as ancilla,system.
    assert np.isclose(ideal.joint_probabilities["00"], probs[0], atol=1e-12)
    assert np.isclose(ideal.joint_probabilities["10"], probs[1], atol=1e-12)
    assert np.isclose(ideal.joint_probabilities["01"], probs[2], atol=1e-12)
    assert np.isclose(ideal.joint_probabilities["11"], probs[3], atol=1e-12)
    assert np.isclose(
        ideal.ancilla_success_probability,
        probs[0] + probs[1],
        atol=1e-12,
    )


def test_helios_configs_target_the_right_devices():
    from hardware.quantinuum_runner import (
        helios_emulator,
        helios_emulator_lite,
        helios_hardware,
        helios_syntax_checker,
    )

    # Quantinuum-hosted devices are targeted by system_name.
    assert helios_syntax_checker().system_name == "Helios-1SC"
    assert helios_emulator().system_name == "Helios-1E"
    assert helios_hardware().system_name == "Helios-1"
    # Helios-1E-lite is a Nexus-hosted Selene emulator (different backend), with
    # the Helios runtime and the *noisy* QSystem error model.
    lite = helios_emulator_lite()
    assert type(lite).__name__ == "SelenePlusConfig"
    assert type(lite.error_model).__name__ == "QSystemErrorModel"
    assert type(lite.runtime).__name__ == "HeliosRuntime"
