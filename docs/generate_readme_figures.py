"""Generate the README figures for the QSVT linear-solver project.

Figures are built from two sources:

* ``docs/results/readme_results.json`` -- the captured solver sweeps and the
  *measured* hardware/emulator counts (these cannot be recomputed, so they are
  read, never regenerated here).
* ``primitives.qsp_qsvt`` -- the bounded ``1/x`` polynomial, computed live for
  the "what QSVT does" illustration.

Run ``python docs/generate_readme_figures.py`` to refresh the PNGs in
``docs/figures/``.  The data JSON is produced by ``collect_results`` in the
project history; this script only consumes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager  # noqa: F401  (ensures font cache is built)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "docs" / "figures"
DATA = ROOT / "docs" / "results" / "readme_results.json"

# --------------------------------------------------------------------------- #
# House style
# --------------------------------------------------------------------------- #
INK = "#1f2933"          # near-black for text
MUTED = "#7b8794"        # secondary text / grid
IDEAL = "#52606d"        # reference (statevector ideal)
TEAL = "#2a9d8f"         # Quantinuum emulator
CORAL = "#e8804d"        # Quantinuum hardware
BLUE = "#3b6fd6"         # IBM hardware
ACCENT = "#b8002e"       # highlight / error

PLATFORM_COLOR = {
    "Noiseless circuit ideal": IDEAL,
    "Quantinuum Helios-1E-lite": TEAL,
    "Quantinuum Helios-1": CORAL,
    "IBM Kingston": BLUE,
}
SHORT = {
    "Noiseless circuit ideal": "Noiseless\nideal",
    "Quantinuum Helios-1E-lite": "Helios-1E-lite\n(emulator)",
    "Quantinuum Helios-1": "Helios-1\n(hardware)",
    "IBM Kingston": "IBM Kingston\n(hardware)",
}


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": INK,
            "axes.edgecolor": "#c9d2dd",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#e6eaef",
            "grid.linewidth": 1.0,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "legend.frameon": False,
        }
    )


def _title(ax, title, subtitle=None):
    pad = 30 if subtitle else 12
    ax.set_title(title, fontsize=15, fontweight="bold", pad=pad, loc="left")
    if subtitle:
        ax.text(
            0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5,
            color=MUTED, ha="left", va="bottom",
        )


def _despine(ax, left=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def load() -> dict:
    return json.loads(DATA.read_text())


# --------------------------------------------------------------------------- #
# Figure 1 -- the idea: QSVT applies a bounded polynomial approximation of 1/x
# --------------------------------------------------------------------------- #
def fig_polynomial() -> None:
    from primitives.qsp_qsvt import approximate_inverse

    kappa = 5.0
    poly = approximate_inverse(kappa, epsilon=0.05)
    xs = np.linspace(-1, 1, 1600)
    target = np.where(np.abs(xs) >= 1 / kappa, poly.scale / xs, np.nan)
    approx = poly(xs)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    # valid spectral bands (where the approximation is trusted)
    for lo, hi in [(1 / kappa, 1.0), (-1.0, -1 / kappa)]:
        ax.axvspan(lo, hi, color=TEAL, alpha=0.10, lw=0)
    ax.axhline(0, color="#c9d2dd", lw=1)
    for yb in (1.0, -1.0):
        ax.axhline(yb, color=MUTED, lw=1, ls=(0, (1, 3)), alpha=0.7)
    ax.text(-0.98, 1.02, r"block-encodable bound  $|P|\leq 1$",
            color=MUTED, fontsize=9, va="bottom")

    # one polynomial, one target -- the polynomial hugs 1/x inside the bands
    ax.plot(xs, approx, color=BLUE, lw=2.9, zorder=2,
            label=f"QSVT polynomial  P(x)   (degree {poly.degree})")
    ax.plot(xs, target, color=ACCENT, lw=2.4, ls=(0, (6, 3)), zorder=3,
            label=r"target  $\mathrm{scale}\,/\,x$")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("eigenvalue  x   (normalised spectrum of A)")
    ax.set_ylabel("polynomial value")
    ax.text(1 / kappa, -1.13, r"$1/\kappa$", color=TEAL, fontsize=11, ha="center")
    ax.text(-1 / kappa, -1.13, r"$-1/\kappa$", color=TEAL, fontsize=11, ha="center")
    ax.text(0.6, -0.5, "approximation\nvalid here", color=TEAL, fontsize=9.5,
            ha="center", va="center")
    _despine(ax, left=True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", fontsize=10.5, ncols=2,
              bbox_to_anchor=(0.5, -0.30))
    _title(
        ax,
        "QSVT inverts a matrix by approximating 1/x",
        f"A bounded degree-{poly.degree} polynomial hugs 1/x on the well-conditioned "
        f"band (kappa = {kappa:.0f}) and stays within +/-1 elsewhere.",
    )
    fig.subplots_adjust(top=0.86, bottom=0.22)
    fig.savefig(FIG_DIR / "qsvt_polynomial.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 -- classical accuracy: QSVT vs HHL
# --------------------------------------------------------------------------- #
def fig_accuracy(results: dict) -> None:
    qsvt = results["qsvt_sweep"]
    hhl = results["hhl_sweep"]

    fig, (axq, axh) = plt.subplots(1, 2, figsize=(11.6, 4.5))

    axq.plot([r["degree"] for r in qsvt], [r["residual"] for r in qsvt],
             marker="o", ms=8, lw=2.6, color=BLUE, mec="white", mew=1.4)
    axq.set_yscale("log")
    axq.set_xlabel("QSVT polynomial degree")
    axq.set_ylabel(r"relative residual  $\|Ax-b\|/\|b\|$")
    for r in qsvt:
        axq.annotate(f"{r['residual']:.0e}", (r["degree"], r["residual"]),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=8.5, color=MUTED, ha="center")
    _title(axq, "QSVT accuracy is tunable",
           "Higher polynomial degree -> exponentially smaller error.")

    axh.plot([r["clock_qubits"] for r in hhl], [r["residual"] for r in hhl],
             marker="s", ms=8, lw=2.6, color=CORAL, mec="white", mew=1.4)
    axh.set_yscale("log")
    axh.set_xlabel("HHL phase-estimation clock qubits")
    axh.set_ylabel(r"relative residual  $\|Ax-b\|/\|b\|$")
    _title(axh, "HHL baseline for comparison",
           "Accuracy set by QPE resolution; non-monotone on the finite grid.")

    for ax in (axq, axh):
        _despine(ax, left=True)
        ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "solver_convergence.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 -- the headline hardware result (the recovered solution)
# --------------------------------------------------------------------------- #
def fig_hardware_solution(results: dict) -> None:
    hw = results["hardware"]
    order = [
        "Noiseless circuit ideal",
        "Quantinuum Helios-1E-lite",
        "Quantinuum Helios-1",
        "IBM Kingston",
    ]
    p1 = [hw[k]["postselected_system_probabilities"]["1"] for k in order]
    ideal1 = hw["Noiseless circuit ideal"]["postselected_system_probabilities"]["1"]

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    y = np.arange(len(order))[::-1]  # ideal on top
    ax.set_ylim(-0.8, len(order) - 0.4)

    # ideal reference band (+/- 2 pp)
    ax.axvspan((ideal1 - 0.02) * 100, (ideal1 + 0.02) * 100,
               color=IDEAL, alpha=0.10, lw=0)
    ax.axvline(ideal1 * 100, color=IDEAL, lw=1.8, ls=(0, (5, 3)), zorder=1)
    ax.text(ideal1 * 100 + 0.45, -0.72, f"ideal  {ideal1:.1%}",
            color=IDEAL, fontsize=10, ha="left", va="center")

    for yi, key in zip(y, order):
        val = hw[key]["postselected_system_probabilities"]["1"] * 100
        color = PLATFORM_COLOR[key]
        ax.plot([0, val], [yi, yi], color=color, lw=3, alpha=0.55, zorder=2)
        ax.scatter([val], [yi], s=170, color=color, zorder=3,
                   edgecolor="white", linewidth=1.6)
        delta = (hw[key]["postselected_system_probabilities"]["1"] - ideal1) * 100
        tag = "" if key == "Noiseless circuit ideal" else f"   (Δ {delta:+.1f} pp)"
        ax.text(val + 0.4, yi, f"{val:.1f}%{tag}", va="center", fontsize=10.5,
                color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[k].replace("\n", " ") for k in order], fontsize=11)
    ax.set_xlim(0, max(p1) * 100 + 7)
    ax.set_xlabel(r"measured probability of system qubit $=|1\rangle$  "
                  r"(after ancilla post-selection)")
    _despine(ax)
    ax.grid(axis="y", visible=False)
    _title(
        ax,
        "The same solution, recovered on three quantum backends",
        "Post-selected system-qubit distribution; closer to the dashed ideal is better.",
    )
    fig.subplots_adjust(top=0.84)
    fig.savefig(FIG_DIR / "hardware_solution.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4 -- where the noise shows up (joint-outcome heatmap)
# --------------------------------------------------------------------------- #
def fig_hardware_fidelity(results: dict) -> None:
    hw = results["hardware"]
    order = [
        "Noiseless circuit ideal",
        "Quantinuum Helios-1E-lite",
        "Quantinuum Helios-1",
        "IBM Kingston",
    ]
    bitstrings = ["00", "10", "01", "11"]
    col_titles = [
        "00\nsolution |0>",
        "10\nsolution |1>",
        "01\nancilla leak",
        "11\nancilla leak",
    ]

    def fracs(key):
        row = hw[key]
        if "joint_probabilities" in row:
            return [row["joint_probabilities"][b] for b in bitstrings]
        shots = row["shots"]
        return [row["joint_counts"].get(b, 0) / shots for b in bitstrings]

    matrix = np.array([fracs(k) for k in order])

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    # perceptual emphasis on the small (error) outcomes via sqrt scaling
    shaded = np.sqrt(matrix)
    im = ax.imshow(shaded, cmap="BuPu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(bitstrings)))
    ax.set_xticklabels(col_titles, fontsize=10.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[k].replace("\n", " ") for k in order], fontsize=10.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(len(order)):
        for j in range(len(bitstrings)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.1%}", ha="center", va="center",
                    fontsize=11, fontweight="bold" if j < 2 else "normal",
                    color="white" if shaded[i, j] > 0.55 else INK)
    # separate signal (00/10) from leakage (01/11)
    ax.axvline(1.5, color="white", lw=4)
    ax.axvline(1.5, color="#c9d2dd", lw=1.2, ls=(0, (4, 3)))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("shot fraction (sqrt-scaled)", color=MUTED, fontsize=9.5)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8.5)

    _title(
        ax,
        "Where each device spends its shots",
        "Left of the divider is the encoded answer; right is ancilla leakage "
        "(noise). The emulator/hardware concentrate on 00/10, as intended.",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hardware_joint_counts.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = load()
    fig_polynomial()
    fig_accuracy(results)
    fig_hardware_solution(results)
    fig_hardware_fidelity(results)
    print("wrote figures to", FIG_DIR)


if __name__ == "__main__":
    main()
