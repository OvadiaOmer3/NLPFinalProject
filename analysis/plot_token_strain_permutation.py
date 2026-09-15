"""Appendix figure: the token-strain permutation null, for the paper.

Fits nothing. Reads ``permutation_reported`` (all draws of the reported permutation
test) from the JSON written by ``analysis/strain_diagnostics.py --permutations N``
and draws the null distribution with the observed statistic marked.

    python analysis/plot_token_strain_permutation.py \\
        --diagnostics data/artifacts/features/strain_diagnostics_specE.json \\
        --out ../token_strain_permutation.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette, light mode: categorical slots 1-2 (validated: CVD dE 24.7,
# normal-vision dE 33.6, contrast >= 3:1). Text uses ink tokens, never series color.
NULL_C, OBS_C = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostics",
                    default="data/artifacts/features/strain_diagnostics_specE.json")
    ap.add_argument("--out", default="../token_strain_permutation.pdf")
    ap.add_argument("--dpi", type=int, default=300, help="raster resolution (only used if --out is a .png)")
    a = ap.parse_args()

    dj = json.loads(Path(a.diagnostics).read_text(encoding="utf-8"))
    if "permutation_reported" not in dj:
        raise SystemExit(f"{a.diagnostics} has no permutation_reported -- rerun "
                         "strain_diagnostics.py with --permutations 150")
    r = dj["permutation_reported"]
    null, obs = np.asarray(r["null"]), float(r["observed"])
    n, ge = len(null), int((null >= obs).sum())

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    bins = np.arange(0, max(null.max(), obs) + 2.0, 2.0)
    ax.hist(null, bins=bins, color=NULL_C, edgecolor="white", linewidth=0.8)
    ax.axvline(obs, color=OBS_C, linewidth=1.6)
    top = ax.get_ylim()[1]
    ax.text(obs + 0.8, top * 0.93,
            f"observed $\\chi^2$ = {obs:.1f}\n$p$ = {r['p_empirical']:.3f} ({ge}/{n})",
            color=INK, va="top", ha="left", fontsize=7.5)
    ax.set_xlabel(r"Likelihood-ratio $\chi^2(2)$ for the strain terms")
    ax.set_ylabel("Permutations")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"wrote {a.out}  (observed {obs:.2f}, {ge}/{n} >= observed, "
          f"p={r['p_empirical']:.3f})")


if __name__ == "__main__":
    main()
