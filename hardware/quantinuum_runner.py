"""Quantinuum (Helios) submission for QSVT circuits.

This turns the *same* block-encoding + QSVT objects used by the solver into a
concrete ``pytket`` circuit, lowers it to HUGR, and runs it through Quantinuum
Nexus in three escalating stages, each a real device (``qnexus.devices.get_all``):

    Helios-1SC  (syntax checker)            <- cheapest pre-flight, no physics
        |
    Helios-1E   (noisy emulator)            <- realistic QSystem error model
        |
    Helios-1    (real hardware)             <- queued; may wait

Each stage is targeted by ``HeliosConfig(system_name=...)``, so the device shown
in Nexus matches the stage.  Note Helios-1E is *noisy by default*
(``QSystemErrorModel``); the emulator is not a noiseless simulator.

Helios is the right target for these circuits: high two-qubit fidelity and
all-to-all connectivity suit the deeper, alternating QSVT sequences.

Authentication
--------------
You authenticate (the agent does not).  Run once in your shell / notebook::

    import qnexus
    qnexus.login()            # opens a browser device-code flow

after which the calls below pick up the cached token.  Nothing here reads your
credential files directly.

Device names
------------
The stages target the devices ``Helios-1SC`` / ``Helios-1E`` / ``Helios-1`` by
``system_name``.  Before a real run, confirm what your account can see::

    import qnexus
    qnexus.devices.get_all()                      # lists devices + status

If Helios-1 is under maintenance the hardware job still *submits* and queues;
use :meth:`NexusRunner.wait` (or poll later) to collect results once it runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from primitives.qsp_qsvt import wx_block_encoding

# ----------------------------------------------------------------------------- #
# Circuit construction (pytket)
# ----------------------------------------------------------------------------- #
def build_qsvt_circuit(
    A: np.ndarray,
    phis: np.ndarray,
    b: np.ndarray,
    measure: bool = True,
):
    """Build the QSVT circuit for a 2x2 Hermitian ``A`` (one system qubit).

    Layout (pytket ILO-BE, qubit 0 most significant):
        q0 = block-encoding ancilla
        q1 = system qubit (holds |b>, then the QSVT-transformed state)

    The circuit prepares ``|b>`` on the system qubit, then applies the
    alternating QSVT sequence ``S(phi_0) prod_k U_A S(phi_k)`` where ``U_A`` is
    the ``W_x`` block-encoding of ``A`` and ``S(phi) = e^{i phi Z}`` on the
    ancilla.  Post-selecting the ancilla on ``|0>`` yields the encoded block
    ``<0| Q_Phi |0> |b>`` (a complex polynomial of ``A`` applied to ``|b>``).

    Returns a ``pytket.circuit.Circuit``.
    """
    from pytket.circuit import Circuit, Op, OpType, StatePreparationBox, Unitary2qBox

    A = np.asarray(A, dtype=complex)
    if A.shape != (2, 2):
        raise ValueError(
            "build_qsvt_circuit currently targets a 2x2 A (1 system qubit); "
            "larger systems need an n-qubit block-encoding box."
        )
    phis = np.asarray(phis, dtype=float)

    U_A = wx_block_encoding(A)  # 4x4, ancilla = most-significant qubit
    be_box = Unitary2qBox(U_A)

    circ = Circuit(2)
    anc, sys = circ.qubits  # q0 = ancilla, q1 = system

    # 1. prepare |b> on the system qubit
    b = np.asarray(b, dtype=complex).reshape(-1)
    b = b / np.linalg.norm(b)
    circ.add_gate(StatePreparationBox(b), [sys])

    # 2. QSVT sequence:  S(phi_0)  then  [ U_A  S(phi_k) ] for k = 1..d
    #    S(phi) = e^{i phi Z} = Rz(-2 phi) up to global phase.
    circ.Rz(-2.0 * phis[0] / np.pi, anc)  # pytket Rz is in half-turns of pi
    for phi in phis[1:]:
        circ.add_unitary2qbox(be_box, anc, sys)
        circ.Rz(-2.0 * phi / np.pi, anc)

    if measure:
        circ.measure_all()
    return circ


def simulate_statevector(circuit) -> np.ndarray:
    """Ideal statevector of a measurement-free circuit (local; no Nexus)."""
    return np.asarray(circuit.get_statevector())


def circuit_to_hugr(circuit):
    """Convert a pytket circuit to a runnable HUGR ``Package`` for Helios.

    The Helios runtime executes HUGR programs whose entry point is a *no-input*
    ``main()`` that allocates its own qubits, runs, measures, and reports
    classical results -- not the qubits-in/qubits-out function that
    ``tket.from_tket1`` produces (which the runtime rejects with "Entry point
    function must have no input parameters").

    So we decompose boxes, rebase to ``{Rz, H, CX}`` (the ops in tket's quantum
    extension), then hand-build ``main()`` with the HUGR builder:
    ``QAlloc`` per qubit -> gates -> ``Measure`` + ``QFree`` -> ``result_bool``
    per qubit (tagged ``c0, c1, ...``).  Helios does native-gate compilation
    server-side.
    """
    import hugr.tys as tket_tys
    import tket.extensions as ext
    from hugr.build.function import Module
    from hugr.package import Package
    from hugr.std.float import FloatVal
    from pytket.circuit import OpType
    from pytket.passes import AutoRebase, DecomposeBoxes

    circ = circuit.copy()
    DecomposeBoxes().apply(circ)
    AutoRebase({OpType.Rz, OpType.H, OpType.CX}).apply(circ)

    n = circ.n_qubits
    qindex = {q: i for i, q in enumerate(circ.qubits)}

    # Concrete ops from tket's quantum/rotation/result extensions (correctly
    # typed -- note Rz takes a `rotation`, reached from half-turns via
    # from_halfturns_unchecked).
    quantum = ext.quantum()
    op_qalloc = quantum.get_op("QAlloc").instantiate([])
    op_qfree = quantum.get_op("QFree").instantiate([])
    op_measure = quantum.get_op("Measure").instantiate([])
    op_h = quantum.get_op("H").instantiate([])
    op_cx = quantum.get_op("CX").instantiate([])
    op_rz = quantum.get_op("Rz").instantiate([])
    op_from_halfturns = ext.rotation().get_op("from_halfturns_unchecked").instantiate([])
    result_bool = ext.result().get_op("result_bool")

    module = Module()
    main = module.define_main([])
    wires = [main.add_op(op_qalloc)[0] for _ in range(n)]

    for cmd in circ.get_commands():
        op_type = cmd.op.type
        qs = [qindex[q] for q in cmd.qubits]
        if op_type == OpType.H:
            wires[qs[0]] = main.add_op(op_h, wires[qs[0]])[0]
        elif op_type == OpType.CX:
            node = main.add_op(op_cx, wires[qs[0]], wires[qs[1]])
            wires[qs[0]], wires[qs[1]] = node[0], node[1]
        elif op_type == OpType.Rz:
            angle = float(cmd.op.params[0])  # half-turns (same as pytket)
            fl = main.load(FloatVal(angle))
            rot = main.add_op(op_from_halfturns, fl)
            wires[qs[0]] = main.add_op(op_rz, wires[qs[0]], rot)[0]
        elif op_type == OpType.Measure:
            continue  # we measure + report explicitly below
        else:
            raise ValueError(f"unsupported gate for Helios HUGR build: {op_type}")

    for i in range(n):
        node = main.add_op(op_measure, wires[i])
        qubit_after, bit = node[0], node[1]
        main.add_op(op_qfree, qubit_after)
        main.add_op(result_bool.instantiate([tket_tys.StringArg(f"c{i}")]), bit)

    main.set_outputs()

    from hugr.std.float import FLOAT_OPS_EXTENSION, FLOAT_TYPES_EXTENSION

    extensions = [
        ext.rotation(),
        ext.futures(),
        ext.qsystem(),
        ext.quantum(),
        ext.result(),
        FLOAT_TYPES_EXTENSION,
        FLOAT_OPS_EXTENSION,
    ]
    return Package(modules=[module.hugr], extensions=extensions)


# ----------------------------------------------------------------------------- #
# Helios backend configurations
# ----------------------------------------------------------------------------- #
# Devices, from `qnexus.devices.get_all()`:
#
#   Helios-1SC      -- syntax/logic checker (HeliosConfig; cheapest pre-flight)
#   Helios-1E-lite  -- Nexus-hosted Selene emulator of Helios, NOISY (SelenePlus)
#   Helios-1E       -- Quantinuum-hosted emulator, NOISY (HeliosConfig + emu cfg)
#   Helios-1        -- real hardware (HeliosConfig)
#
# Note `Helios-1E-lite` (nexus_hosted) is a *different backend* from `Helios-1E`:
# it runs on the Selene HUGR simulator hosted in Nexus, reached via SelenePlusConfig
# with the Helios runtime + QSystem noise -- not a HeliosConfig system_name.
def helios_syntax_checker():
    """Helios-1SC: syntax/logic checker (cheapest pre-flight, no noise model)."""
    from quantinuum_schemas.models.backend_config import HeliosConfig

    return HeliosConfig(system_name="Helios-1SC")


def helios_emulator_lite():
    """Helios-1E-lite: Nexus-hosted Selene emulator of Helios with QSystem noise.

    Lightweight (26-qubit) noisy emulator that runs on Nexus infrastructure (no
    HQC max_cost needed), built on the Selene HUGR simulator with the Helios
    runtime and the realistic ``QSystemErrorModel``.
    """
    from quantinuum_schemas.models.backend_config import SelenePlusConfig
    from quantinuum_schemas.models.emulator_config import (
        HeliosRuntime,
        QSystemErrorModel,
        StatevectorSimulator,
    )

    return SelenePlusConfig(
        simulator=StatevectorSimulator(),
        runtime=HeliosRuntime(),
        error_model=QSystemErrorModel(),
    )


def helios_emulator():
    """Helios-1E: Quantinuum-hosted noisy emulator (QSystem error model).

    Helios emulation requires an explicit ``emulator_config``; the default
    ``HeliosEmulatorConfig`` uses a statevector simulator with the *noisy*
    ``QSystemErrorModel``.
    """
    from quantinuum_schemas.models.backend_config import (
        HeliosConfig,
        HeliosEmulatorConfig,
    )

    return HeliosConfig(
        system_name="Helios-1E", emulator_config=HeliosEmulatorConfig()
    )


def helios_hardware():
    """Helios-1: real hardware."""
    from quantinuum_schemas.models.backend_config import HeliosConfig

    return HeliosConfig(system_name="Helios-1")


# ----------------------------------------------------------------------------- #
# Nexus runner
# ----------------------------------------------------------------------------- #
@dataclass
class StageResult:
    stage: str
    job_ref: Any = None
    counts: dict | None = None
    status: str = "submitted"
    note: str = ""


@dataclass
class NexusRunner:
    """Drive emulator -> hardware submission for a single circuit.

    You must have run ``qnexus.login()`` already.  Typical use::

        runner = NexusRunner()
        runner.upload(circuit)
        runner.run_syntax_check()                 # Helios-1SC pre-flight
        runner.run_emulator(n_shots=500)          # Helios-1E (noisy emulator)
        hw = runner.submit_hardware(n_shots=500)  # async; queues on Helios-1
        runner.wait(hw)                           # collect when it runs
    """

    project_name: str = "qsvt-linear-solver"
    job_label: str = "QSVT solve Ax=b"
    _project: Any = field(default=None, init=False, repr=False)
    _circuit_ref: Any = field(default=None, init=False, repr=False)
    _n_qubits: int = field(default=0, init=False, repr=False)

    # -- session/project -------------------------------------------------- #
    def project(self):
        import qnexus

        if self._project is None:
            self._project = qnexus.projects.get_or_create(name=self.project_name)
        return self._project

    def upload(self, circuit, name: str = "qsvt-circuit"):
        """Convert a pytket circuit to HUGR and upload it for the Helios stack.

        Helios executes HUGR programs; the circuit is lowered via
        :func:`circuit_to_hugr` and uploaded with ``qnexus.hugr.upload``.
        """
        import qnexus

        hugr_package = circuit_to_hugr(circuit)
        self._n_qubits = circuit.n_qubits  # Helios requires n_qubits per job item
        self._circuit_ref = qnexus.hugr.upload(
            hugr_package=hugr_package, name=name, project=self.project()
        )
        return self._circuit_ref

    def _require_circuit(self):
        if self._circuit_ref is None:
            raise RuntimeError("call upload(circuit) before running")
        return self._circuit_ref

    # -- compile ---------------------------------------------------------- #
    def compile(self, backend_config, name: str = "qsvt-compile", opt_level: int = 2):
        """Compile the uploaded circuit to a backend's native gate set (blocking).

        Note: the Helios stack is *not* a standalone compilation target -- it
        compiles the circuit server-side as part of execution.  This method is
        only for backends that expose a separate compile pass (e.g. the legacy
        H-series ``QuantinuumConfig``).  The Helios run/submit methods below skip
        it and pass the circuit straight to ``execute``.
        """
        import qnexus

        compiled = qnexus.compile(
            programs=self._require_circuit(),
            backend_config=backend_config,
            name=name,
            optimisation_level=opt_level,
            project=self.project(),
        )
        return compiled[0]

    # -- pre-flight + emulators (submit, then poll to completion) --------- #
    # Quantinuum-hosted devices (Helios-1SC/1E/1) require a per-job spending cap
    # (max_cost, in HQCs).  The Nexus-hosted Selene emulator (Helios-1E-lite)
    # does not -- pass max_cost=None there.
    def run_syntax_check(
        self, n_shots: int = 100, max_cost: float = 10.0, timeout: float = 1800.0
    ) -> StageResult:
        """Helios-1SC: cheapest pre-flight -- validates the program will run."""
        return self._submit_and_wait(
            helios_syntax_checker(), n_shots, "Helios-1SC (syntax check)",
            max_cost, timeout,
        )

    def run_emulator_lite(
        self, n_shots: int = 500, timeout: float = 1800.0
    ) -> StageResult:
        """Helios-1E-lite: Nexus-hosted Selene noisy emulator (no HQC cost)."""
        return self._submit_and_wait(
            helios_emulator_lite(), n_shots, "Helios-1E-lite (noisy emulator)",
            None, timeout,
        )

    def run_emulator(
        self, n_shots: int = 500, max_cost: float = 20.0, timeout: float = 1800.0
    ) -> StageResult:
        """Helios-1E: Quantinuum-hosted noisy emulator (realistic error model)."""
        return self._submit_and_wait(
            helios_emulator(), n_shots, "Helios-1E (noisy emulator)", max_cost, timeout,
        )

    def _submit_and_wait(
        self, backend_config, n_shots, stage, max_cost, timeout
    ) -> StageResult:
        """Async-submit then poll to completion.

        We do NOT use the blocking ``qnexus.execute`` (it gives up after its
        300 s default while a job sits in the emulator queue).  Instead we submit
        asynchronously and poll the status so a queued job is waited out.
        """
        import time

        import qnexus

        # Helios compiles server-side at execution time -- submit the program ref
        # directly (no separate compile job).  max_cost only applies to
        # Quantinuum-hosted devices; the Nexus-hosted Selene emulator omits it.
        submit_kwargs = dict(
            programs=self._require_circuit(),
            n_shots=n_shots,
            backend_config=backend_config,
            name=f"{self.job_label} | {stage}",
            project=self.project(),
            n_qubits=self._n_qubits,
        )
        if max_cost is not None:
            submit_kwargs["max_cost"] = max_cost
        job = qnexus.start_execute_job(**submit_kwargs)
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = str(qnexus.jobs.status(job).status)
            if "COMPLETED" in status:
                refs = qnexus.jobs.results(job)
                # The syntax checker completes without shot results -- a pass is
                # the signal that the program is valid for the device pipeline.
                if not refs:
                    return StageResult(
                        stage=stage, job_ref=job, status="completed",
                        note="validated (no shot results)",
                    )
                res = refs[0].download_result()
                return StageResult(
                    stage=stage, job_ref=job, counts=_counts_to_dict(res),
                    status="completed",
                )
            if "ERROR" in status or "CANCELLED" in status:
                detail = getattr(qnexus.jobs.status(job), "error_detail", "")
                return StageResult(
                    stage=stage, job_ref=job, status="error", note=str(detail),
                )
            time.sleep(15)
        return StageResult(
            stage=stage, job_ref=job, status="timeout",
            note="still queued/running; collect later with runner.wait(stage)",
        )

    # -- hardware (async; queued) ---------------------------------------- #
    def submit_hardware(
        self, n_shots: int = 500, max_cost: float = 50.0
    ) -> StageResult:
        """Submit to Helios-1 real hardware asynchronously and return the job ref.

        Does not block: if Helios-1 is under maintenance the job queues and runs
        when the device is back.  Collect later with :meth:`wait`.

        ``max_cost`` (HQCs) is the hard spending cap Quantinuum enforces per job;
        the job is rejected if the estimated cost exceeds it.  This is REAL
        credits -- set it deliberately.
        """
        import qnexus

        backend_config = helios_hardware()
        # Helios compiles server-side; submit the circuit ref directly.
        job = qnexus.start_execute_job(
            programs=self._require_circuit(),
            n_shots=n_shots,
            backend_config=backend_config,
            name=f"{self.job_label} | Helios-1 (hardware)",
            project=self.project(),
            max_cost=max_cost,
            n_qubits=self._n_qubits,
        )
        return StageResult(
            stage="Helios-1 (hardware)",
            job_ref=job,
            status="queued",
            note="queued on Helios-1; call runner.wait(result) to block for it",
        )

    def wait(self, stage: StageResult, timeout: float | None = None) -> StageResult:
        """Block until a queued job completes, then attach its counts."""
        import qnexus

        if stage.job_ref is None:
            raise ValueError("stage has no job_ref to wait on")
        qnexus.jobs.wait_for(stage.job_ref, timeout=timeout)
        result_refs = qnexus.jobs.results(stage.job_ref)
        stage.counts = (
            _counts_to_dict(result_refs[0].download_result()) if result_refs else {}
        )
        stage.status = "completed"
        return stage

    def hardware_status(self, stage: StageResult) -> str:
        """Current status string of a queued job (non-blocking)."""
        import qnexus

        if stage.job_ref is None:
            return stage.status
        return str(qnexus.jobs.status(stage.job_ref).status)


def _counts_to_dict(result) -> dict:
    """Convert a result to a plain counts dict.

    Handles both a pytket ``BackendResult`` (legacy path) and the HUGR
    ``QsysResult`` returned by the Helios stack (per-register counts keyed by the
    ``result_bool`` tags ``c0, c1, ...``).
    """
    # HUGR QsysResult (Helios): per-register Counters over the bool tags.
    if hasattr(result, "register_counts"):
        try:
            reg = result.register_counts()
            return {name: dict(counter) for name, counter in reg.items()}
        except Exception:  # pragma: no cover - depends on result shape
            pass
    # pytket BackendResult (legacy QuantinuumConfig path).
    if hasattr(result, "get_counts"):
        try:
            counts = result.get_counts()
            return {"".join(map(str, k)): int(v) for k, v in counts.items()}
        except Exception:  # pragma: no cover
            pass
    return {}


# ----------------------------------------------------------------------------- #
# Demo entry point
# ----------------------------------------------------------------------------- #
def _demo():
    """Build a small QSVT circuit and run the Helios escalation.

    Run *after* ``qnexus.login()``.  Uses a 2x2 system so the circuit is shallow
    enough for a quick hardware check.
    """
    from primitives.qsp_qsvt import approximate_inverse, find_phases

    # Small, well-conditioned 2x2 Hermitian system.
    A = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=complex)  # eigenvalues 0.5, 1.0
    b = np.array([1.0, 0.0], dtype=complex)

    poly = approximate_inverse(kappa=2.0, epsilon=0.3)  # low degree -> shallow
    phis = find_phases(poly.cheb_coeffs, poly.parity)
    circ = build_qsvt_circuit(A, phis, b, measure=True)
    print(f"QSVT circuit: {circ.n_qubits} qubits, depth {circ.depth()}, "
          f"polynomial degree {poly.degree}")

    runner = NexusRunner()
    runner.upload(circ, name="qsvt-2x2-inverse")

    print("\n[1/3] Helios-1SC (syntax check) ...")
    sc = runner.run_syntax_check(n_shots=100)
    print(f"   {sc.status}: {sc.counts}")

    print("\n[2/3] Helios-1E-lite (Nexus-hosted noisy emulator) ...")
    emu = runner.run_emulator_lite(n_shots=500)
    print(f"   {emu.status}: counts={emu.counts}")

    print("\n[3/3] Helios-1 (real hardware) -- submitting ...")
    hw = runner.submit_hardware(n_shots=500)
    print(f"   {hw.status}: {hw.note}")
    print("   poll with runner.hardware_status(hw); block with runner.wait(hw)")
    return runner, hw


if __name__ == "__main__":  # pragma: no cover
    _demo()
