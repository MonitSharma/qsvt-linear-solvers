"""IBM Quantum hardware submission for the 2-qubit QSVT demo.

This runner targets IBM Quantum Runtime Sampler V2.  It builds the same
block-encoding + QSVT sequence as the Quantinuum path, submits the measured
2-qubit circuit to an IBM backend, and stores joint counts so the ancilla
success branch can be post-selected cleanly.

Credentials are read from environment variables; do not put tokens in this file:

    export IBM_QUANTUM_TOKEN="..."
    export IBM_QUANTUM_CRN="crn:v1:..."

Typical use:

    python -m hardware.run_ibm list-backends
    python -m hardware.run_ibm submit --backend ibm_kingston --shots 2048
    python -m hardware.run_ibm status --job-id <job-id>
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from primitives.qsp_qsvt import (
    approximate_inverse,
    find_phases,
    qsvt_sequence,
    wx_block_encoding,
)


DEFAULT_BACKEND_ORDER = ("ibm_kingston", "ibm_fez", "ibm_marrakesh")
DEFAULT_RESULTS_DIR = Path("hardware/ibm_results")


@dataclass
class CircuitIdeal:
    """Noiseless probabilities for the exact circuit sent to IBM."""

    joint_probabilities: dict[str, float]
    raw_system_probabilities: dict[str, float]
    ancilla_success_probability: float
    postselected_system_probabilities: dict[str, float]


@dataclass
class IBMSubmission:
    """Metadata for a submitted IBM Runtime job."""

    job_id: str
    backend: str
    shots: int
    created_utc: str
    ideal: CircuitIdeal
    status: str = "submitted"
    counts: dict[str, int] | None = None
    postselected_counts: dict[str, int] | None = None
    postselected_probabilities: dict[str, float] | None = None


def demo_problem(verbose: bool = False):
    """Return the 2x2 system used by the hardware demo."""

    A = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=complex)
    b = np.array([1.0, 0.0], dtype=complex)
    if verbose:
        poly = approximate_inverse(kappa=2.0, epsilon=0.3)
        phis = find_phases(poly.cheb_coeffs, poly.parity)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            poly = approximate_inverse(kappa=2.0, epsilon=0.3)
            phis = find_phases(poly.cheb_coeffs, poly.parity)
    return A, b, phis


def build_qiskit_qsvt_circuit(A: np.ndarray, b: np.ndarray, phis: np.ndarray):
    """Build the IBM/Qiskit version of the 2-qubit QSVT circuit.

    Qubit 0 is the block-encoding ancilla and qubit 1 is the system qubit.
    Measurements are stored in the two-bit classical register ``c`` with
    ``c[0] <- ancilla`` and ``c[1] <- system``.  Qiskit count keys display
    classical bits high-to-low, so key ``"10"`` means system=1, ancilla=0.
    """

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, UnitaryGate

    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex).reshape(-1)
    phis = np.asarray(phis, dtype=float)

    if A.shape != (2, 2):
        raise ValueError("build_qiskit_qsvt_circuit currently supports a 2x2 A")
    if b.shape != (2,):
        raise ValueError("build_qiskit_qsvt_circuit currently supports a 2-entry b")

    b = b / np.linalg.norm(b)
    # The local matrix code and pytket runner use big-endian basis
    # |ancilla, system>.  Qiskit's statevector/count keys are little-endian for
    # q0/q1, so conjugate the 2-qubit block by the SWAP permutation before
    # attaching it to qargs [ancilla, system].
    swap_basis = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )
    block_encoding = UnitaryGate(
        swap_basis @ wx_block_encoding(A) @ swap_basis,
        label="W_x(A)",
    )

    qc = QuantumCircuit(2, 2, name="qsvt_linear_solve")
    anc, sys = 0, 1

    qc.append(StatePreparation(b), [sys])
    qc.rz(-2.0 * float(phis[0]), anc)
    for phi in phis[1:]:
        qc.append(block_encoding, [anc, sys])
        qc.rz(-2.0 * float(phi), anc)

    qc.measure(anc, 0)
    qc.measure(sys, 1)
    return qc


def circuit_ideal(A: np.ndarray, b: np.ndarray, phis: np.ndarray) -> CircuitIdeal:
    """Compute noiseless probabilities for the exact pre-measurement circuit."""

    from qiskit.quantum_info import Statevector

    qc = build_qiskit_qsvt_circuit(A, b, phis).remove_final_measurements(inplace=False)
    probs = Statevector.from_instruction(qc).probabilities_dict()

    # Qiskit keys are qubit-1 then qubit-0 for a 2-qubit state.  With our layout
    # that is system then ancilla, matching Sampler count-key display.
    joint = {key: float(probs.get(key, 0.0)) for key in ("00", "01", "10", "11")}
    raw_system = {
        "0": joint["00"] + joint["01"],
        "1": joint["10"] + joint["11"],
    }
    success = joint["00"] + joint["10"]
    post = {
        "0": joint["00"] / success if success else 0.0,
        "1": joint["10"] / success if success else 0.0,
    }
    return CircuitIdeal(
        joint_probabilities=joint,
        raw_system_probabilities=raw_system,
        ancilla_success_probability=float(success),
        postselected_system_probabilities=post,
    )


def matrix_qsvt_ideal(A: np.ndarray, b: np.ndarray, phis: np.ndarray) -> dict[str, Any]:
    """Return the matrix-model ideal, useful as an independent sanity check."""

    b = np.asarray(b, dtype=complex).reshape(-1)
    b = b / np.linalg.norm(b)
    Q = qsvt_sequence(wx_block_encoding(A), phis)
    state = Q @ np.kron([1, 0], b)
    probs = np.abs(state) ** 2
    success = float(probs[0] + probs[1])
    return {
        "raw": {
            "00": float(probs[0]),
            "01": float(probs[1]),
            "10": float(probs[2]),
            "11": float(probs[3]),
        },
        "ancilla_success_probability": success,
        "postselected_system_probabilities": {
            "0": float(probs[0] / success),
            "1": float(probs[1] / success),
        },
    }


def service():
    """Create a Qiskit Runtime service from environment variables."""

    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_QUANTUM_CRN") or os.environ.get("IBM_QUANTUM_INSTANCE")
    if not token:
        raise RuntimeError("set IBM_QUANTUM_TOKEN before using the IBM runner")
    if not instance:
        raise RuntimeError("set IBM_QUANTUM_CRN before using the IBM runner")
    return QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )


def list_backends() -> list[dict[str, Any]]:
    """List available real IBM backends with queue/status details."""

    svc = service()
    rows = []
    for backend in svc.backends(simulator=False, operational=True):
        status = backend.status()
        rows.append(
            {
                "name": backend.name,
                "num_qubits": getattr(backend, "num_qubits", None),
                "pending_jobs": getattr(status, "pending_jobs", None),
                "status_msg": getattr(status, "status_msg", ""),
            }
        )
    return sorted(rows, key=lambda row: (row["pending_jobs"] or 0, row["name"]))


def choose_backend(preferred: str | None = None):
    """Choose an operational backend, preferring the shallow-run candidates."""

    svc = service()
    if preferred:
        return svc.backend(preferred)

    available = {backend.name: backend for backend in svc.backends(simulator=False, operational=True)}
    for name in DEFAULT_BACKEND_ORDER:
        if name in available:
            return available[name]
    if not available:
        raise RuntimeError("no operational non-simulator IBM backends are available")
    return sorted(
        available.values(),
        key=lambda backend: (backend.status().pending_jobs, backend.name),
    )[0]


def transpile_for_backend(circuit, backend, optimization_level: int = 3):
    """Map and optimize the circuit for the selected IBM backend."""

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
    )
    return pm.run(circuit)


def counts_from_result(result) -> dict[str, int]:
    """Extract Sampler V2 counts from a single-circuit result."""

    pub_result = result[0]
    data = pub_result.data
    if hasattr(data, "c"):
        return {str(key): int(value) for key, value in data.c.get_counts().items()}
    if hasattr(data, "meas"):
        return {str(key): int(value) for key, value in data.meas.get_counts().items()}
    # Fallback for unusual classical register names.
    for value in data.values():
        if hasattr(value, "get_counts"):
            return {str(key): int(count) for key, count in value.get_counts().items()}
    raise ValueError(f"could not find bitstring counts in result data fields: {list(data.keys())}")


def postselect_ancilla_zero(counts: dict[str, int]) -> tuple[dict[str, int], dict[str, float]]:
    """Post-select count keys where ancilla=0 and return system counts/probs."""

    selected = {"0": 0, "1": 0}
    for key, count in counts.items():
        bitstring = str(key).replace(" ", "")
        if len(bitstring) < 2:
            continue
        system, ancilla = bitstring[-2], bitstring[-1]
        if ancilla == "0":
            selected[system] += int(count)

    total = selected["0"] + selected["1"]
    probs = {
        "0": selected["0"] / total if total else 0.0,
        "1": selected["1"] / total if total else 0.0,
    }
    return selected, probs


def save_submission(record: IBMSubmission, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    """Persist submission metadata/counts as JSON."""

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{record.created_utc.replace(':', '').replace('-', '')}_{record.backend}_{record.job_id}.json"
    path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n")
    return path


def submit(
    backend_name: str | None,
    shots: int,
    optimization_level: int,
    wait: bool,
    results_dir: Path,
) -> IBMSubmission:
    """Submit the QSVT circuit to IBM Runtime Sampler V2."""

    from qiskit_ibm_runtime import SamplerV2 as Sampler

    A, b, phis = demo_problem()
    ideal = circuit_ideal(A, b, phis)
    backend = choose_backend(backend_name)
    circuit = build_qiskit_qsvt_circuit(A, b, phis)
    isa_circuit = transpile_for_backend(circuit, backend, optimization_level)

    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=shots)

    record = IBMSubmission(
        job_id=job.job_id(),
        backend=backend.name,
        shots=shots,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ideal=ideal,
    )

    if wait:
        result = job.result()
        record.counts = counts_from_result(result)
        post_counts, post_probs = postselect_ancilla_zero(record.counts)
        record.postselected_counts = post_counts
        record.postselected_probabilities = post_probs
        record.status = str(job.status())

    save_submission(record, results_dir)
    return record


def fetch_status(job_id: str, results_dir: Path, wait: bool = False) -> IBMSubmission:
    """Fetch status and counts for an existing IBM Runtime job."""

    svc = service()
    job = svc.job(job_id)
    A, b, phis = demo_problem()
    record = IBMSubmission(
        job_id=job.job_id(),
        backend=job.backend().name,
        shots=0,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ideal=circuit_ideal(A, b, phis),
        status=str(job.status()),
    )

    if wait or "DONE" in record.status:
        result = job.result()
        record.counts = counts_from_result(result)
        record.shots = sum(record.counts.values())
        post_counts, post_probs = postselect_ancilla_zero(record.counts)
        record.postselected_counts = post_counts
        record.postselected_probabilities = post_probs
        record.status = str(job.status())

    save_submission(record, results_dir)
    return record


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-backends", help="list available operational IBM backends")

    ideal_parser = sub.add_parser("ideal", help="print noiseless circuit ideal")
    ideal_parser.add_argument("--matrix-check", action="store_true")

    submit_parser = sub.add_parser("submit", help="submit the IBM hardware job")
    submit_parser.add_argument("--backend", default=None)
    submit_parser.add_argument("--shots", type=int, default=2048)
    submit_parser.add_argument("--optimization-level", type=int, default=3)
    submit_parser.add_argument("--wait", action="store_true")
    submit_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    status_parser = sub.add_parser("status", help="fetch job status/counts")
    status_parser.add_argument("--job-id", required=True)
    status_parser.add_argument("--wait", action="store_true")
    status_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    args = parser.parse_args(argv)

    if args.command == "list-backends":
        _print_json(list_backends())
        return 0

    if args.command == "ideal":
        A, b, phis = demo_problem()
        payload: dict[str, Any] = {"circuit": asdict(circuit_ideal(A, b, phis))}
        if args.matrix_check:
            payload["matrix_qsvt"] = matrix_qsvt_ideal(A, b, phis)
        _print_json(payload)
        return 0

    if args.command == "submit":
        record = submit(
            backend_name=args.backend,
            shots=args.shots,
            optimization_level=args.optimization_level,
            wait=args.wait,
            results_dir=args.results_dir,
        )
        _print_json(asdict(record))
        return 0

    if args.command == "status":
        record = fetch_status(args.job_id, args.results_dir, wait=args.wait)
        _print_json(asdict(record))
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
