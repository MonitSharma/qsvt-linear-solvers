"""Generate README figures for the QSVT linear-solver project."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware.run_ibm import circuit_ideal, demo_problem
from solvers.hhl_baseline import hhl_solve
from solvers.qsvt_linear_solver import solve


FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "docs" / "results"


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _spd(n: int, seed: int, shift: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return mat @ mat.conj().T + shift * np.eye(n)


def _postselected_probs(joint_counts: dict[str, int]) -> dict[str, float]:
    selected = {"0": 0, "1": 0}
    for key, count in joint_counts.items():
        system, ancilla = key[0], key[1]
        if ancilla == "0":
            selected[system] += count
    total = selected["0"] + selected["1"]
    return {key: value / total for key, value in selected.items()}


def collect_results() -> dict:
    """Collect deterministic simulation metrics and measured hardware data."""

    A = _spd(4, seed=11).real
    A = (A + A.T) / 2
    rng = np.random.default_rng(12)
    b = rng.standard_normal(4)

    qsvt_sweep = []
    for eps in [0.3, 0.1, 0.03, 0.01, 0.003]:
        result = _quiet(solve, A, b, epsilon=eps)
        qsvt_sweep.append(
            {
                "epsilon": eps,
                "degree": result.degree,
                "residual": result.residual,
                "success_probability": result.success_probability,
            }
        )

    hhl_sweep = []
    for clock_qubits in range(4, 10):
        result = hhl_solve(A, b, clock_qubits=clock_qubits)
        hhl_sweep.append(
            {
                "clock_qubits": clock_qubits,
                "residual": result.residual,
                "success_probability": result.success_probability,
            }
        )

    A_cmp = _spd(4, seed=21).real
    A_cmp = (A_cmp + A_cmp.T) / 2
    rng_cmp = np.random.default_rng(22)
    b_cmp = rng_cmp.standard_normal(4)
    x_true = np.linalg.solve(A_cmp, b_cmp)
    qsvt_cmp = _quiet(solve, A_cmp, b_cmp, epsilon=0.01)
    hhl_cmp = hhl_solve(A_cmp, b_cmp, clock_qubits=9)

    A_hw, b_hw, phis = demo_problem()
    ideal = circuit_ideal(A_hw, b_hw, phis)

    hardware = {
        "Noiseless circuit ideal": {
            "kind": "statevector",
            "shots": None,
            "joint_probabilities": ideal.joint_probabilities,
            "raw_system_probabilities": ideal.raw_system_probabilities,
            "ancilla_success_probability": ideal.ancilla_success_probability,
            "postselected_system_probabilities": ideal.postselected_system_probabilities,
        },
        "Quantinuum Helios-1E-lite": {
            "kind": "noisy emulator",
            "job_id": "65245a1e-20b6-49f3-8265-787684c59298",
            "result_id": "c5fa1063-a11d-4875-8450-bdfca1cbabe3",
            "shots": 100,
            "joint_counts": {"00": 81, "01": 1, "10": 17, "11": 1},
            "postselected_system_probabilities": _postselected_probs(
                {"00": 81, "01": 1, "10": 17, "11": 1}
            ),
        },
        "Quantinuum Helios-1": {
            "kind": "hardware",
            "job_id": "02e36ce1-cf5b-4510-8f7f-6d9afac3c8bb",
            "result_id": "f5deb306-1304-4478-ac62-b8b6c587b6f8",
            "shots": 500,
            "cost_hqc": 49.61,
            "joint_counts": {"00": 432, "01": 17, "10": 37, "11": 14},
            "postselected_system_probabilities": _postselected_probs(
                {"00": 432, "01": 17, "10": 37, "11": 14}
            ),
        },
        "IBM Kingston": {
            "kind": "hardware",
            "job_id": "d8r3pvekodhs7383v6ag",
            "shots": 1024,
            "joint_counts": {"00": 841, "01": 28, "10": 108, "11": 47},
            "postselected_system_probabilities": _postselected_probs(
                {"00": 841, "01": 28, "10": 108, "11": 47}
            ),
        },
    }

    return {
        "classical_comparison": {
            "qsvt": {
                "epsilon": 0.01,
                "degree": qsvt_cmp.degree,
                "kappa": qsvt_cmp.kappa,
                "residual": qsvt_cmp.residual,
                "relative_solution_error": float(
                    np.linalg.norm(qsvt_cmp.x - x_true) / np.linalg.norm(x_true)
                ),
                "success_probability": qsvt_cmp.success_probability,
                "amplification_rounds": qsvt_cmp.amplification_rounds,
            },
            "hhl": {
                "clock_qubits": 9,
                "residual": hhl_cmp.residual,
                "relative_solution_error": float(
                    np.linalg.norm(hhl_cmp.x - x_true) / np.linalg.norm(x_true)
                ),
                "success_probability": hhl_cmp.success_probability,
            },
        },
        "qsvt_sweep": qsvt_sweep,
        "hhl_sweep": hhl_sweep,
        "hardware": hardware,
    }


def plot_solver_convergence(results: dict) -> None:
    qsvt = results["qsvt_sweep"]
    hhl = results["hhl_sweep"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180)
    fig.patch.set_facecolor("white")

    axes[0].plot(
        [row["degree"] for row in qsvt],
        [row["residual"] for row in qsvt],
        marker="o",
        linewidth=2.4,
        color="#2457A6",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("QSVT residual vs polynomial degree")
    axes[0].set_xlabel("QSVT polynomial degree")
    axes[0].set_ylabel(r"Relative residual $\|Ax-b\|/\|b\|$")

    axes[1].plot(
        [row["clock_qubits"] for row in hhl],
        [row["residual"] for row in hhl],
        marker="s",
        linewidth=2.4,
        color="#8F3D2E",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("HHL residual vs QPE clock qubits")
    axes[1].set_xlabel("Clock qubits")
    axes[1].set_ylabel(r"Relative residual $\|Ax-b\|/\|b\|$")

    for ax in axes:
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.12)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Classical statevector validation on a 4x4 SPD system", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "solver_convergence.png", bbox_inches="tight")
    plt.close(fig)


def plot_hardware_postselection(results: dict) -> None:
    hardware = results["hardware"]
    labels = list(hardware)
    p0 = [
        hardware[label]["postselected_system_probabilities"]["0"]
        for label in labels
    ]
    p1 = [
        hardware[label]["postselected_system_probabilities"]["1"]
        for label in labels
    ]

    display_labels = [
        "Ideal",
        "Quantinuum\nHelios-1E-lite",
        "Quantinuum\nHelios-1",
        "IBM\nKingston",
    ]
    x = np.arange(len(labels))
    width = 0.34

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.bar(x - width / 2, p0, width, label="system |0>", color="#2457A6")
    ax.bar(x + width / 2, p1, width, label="system |1>", color="#D39C2F")
    ax.axhline(
        hardware["Noiseless circuit ideal"]["postselected_system_probabilities"]["0"],
        color="#2457A6",
        linewidth=1.2,
        linestyle="--",
        alpha=0.45,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability after ancilla=0 post-selection")
    ax.set_title("Hardware agreement with the QSVT circuit ideal")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    for i, value in enumerate(p0):
        ax.text(i - width / 2, value + 0.025, f"{value:.1%}", ha="center", fontsize=9)
    for i, value in enumerate(p1):
        ax.text(i + width / 2, value + 0.025, f"{value:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hardware_postselection.png", bbox_inches="tight")
    plt.close(fig)


def plot_joint_counts(results: dict) -> None:
    hardware = {
        label: row
        for label, row in results["hardware"].items()
        if "joint_counts" in row
    }
    bitstrings = ["00", "10", "01", "11"]
    colors = ["#2457A6", "#D39C2F", "#6BAA75", "#8F3D2E"]
    labels = list(hardware)
    display_labels = ["Helios-1E-lite", "Helios-1", "IBM Kingston"]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 4.5), dpi=180)
    fig.patch.set_facecolor("white")

    for bitstring, color in zip(bitstrings, colors):
        values = [
            hardware[label]["joint_counts"].get(bitstring, 0) / hardware[label]["shots"]
            for label in labels
        ]
        ax.bar(x, values, bottom=bottom, label=bitstring, color=color)
        bottom += np.asarray(values)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of shots")
    ax.set_title("Measured joint outcomes (key order: system, ancilla)")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.legend(title="bitstring", frameon=False, ncols=4, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hardware_joint_counts.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = collect_results()
    (DATA_DIR / "readme_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    plot_solver_convergence(results)
    plot_hardware_postselection(results)
    plot_joint_counts(results)


if __name__ == "__main__":
    main()
