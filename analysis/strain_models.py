"""Fit the forgetting models over the strain panel and print the comparison.

The question this answers: does token interference explain forgetting *on top of*
how much the model saw the fact?  So every feature family is scored as an
addition to an **exposure-only baseline**, never against an empty model --
`distract`, `strain` and `evidence` are all correlated with plain frequency, and
without that control the interference features mostly re-measure exposure.

The headline contrast is `strain_static` vs `strain_windowed`.  A static feature
cannot explain *when* a fact is forgotten, only which facts are; the windowed
form is the actual hypothesis ("how many competing entities appeared since the
last checkpoint").

Usage:
    python analysis/strain_models.py --panel data/artifacts/features/strain_panel.parquet
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

# Relations too thin to carry their own fixed effect; keeping them makes the
# design matrix singular (perfect separation) rather than merely noisy.
MIN_ROWS, MIN_EVENTS = 100, 20

# --------------------------------------------------------------------------- #
# Model specification -- the single source of truth.  `null_control_distribution`
# and the plotting script both import these, so a change to the baseline can
# never leave one of them fitting a different model than the others.
# --------------------------------------------------------------------------- #
# Exposure control uses BOTH the windowed and the total co-occurrence: how much of
# this fact the model saw during the interval it was at risk in, and overall.
BASE = ("forgot ~ l_evidence_w + l_evidence_static + l_s_pop + l_o_pop "
        "+ C(prop) + C(window)")
# Strain rises steeply with how many tokens a name has, so without this the strain
# terms can be a proxy for name length.  Token count only -- `charlen` measures the
# same construct at spearman 0.88 and splits collinear credit; it stays in the panel
# as a robustness check, out of the reported specification.
NAMELEN = " + l_s_ntok + l_o_ntok"
STRAIN = " + l_subj_strain + l_obj_strain"
STRAIN_W = " + l_subj_strain_w + l_obj_strain_w"
F1 = " + l_s_own_rare + l_o_own_rare + l_s_rivals_rare + l_o_rivals_rare"
F2 = " + ratio_static"
F2_W = " + ratio_w"

MODELS = {
    "exposure only": BASE,
    "+ NAME LENGTH": BASE + NAMELEN,
    "+ strain STATIC (no len ctrl)": BASE + STRAIN,
    "+ strain STATIC | name length": BASE + NAMELEN + STRAIN,
    "+ strain WINDOWED | name length": BASE + NAMELEN + STRAIN_W,
    "+ F1 ownership/rivals | len": BASE + NAMELEN + F1,
    "+ F2 ratio STATIC | len": BASE + NAMELEN + F2,
    "+ F2 ratio WINDOWED | len": BASE + NAMELEN + F2_W,
    "+ everything | name length": BASE + NAMELEN + STRAIN_W + F1 + F2_W,
}


class LR:
    """A likelihood-ratio test of one nested model against another."""

    def __init__(self, chi2: float, df: int, p: float):
        self.chi2, self.df, self.p = chi2, df, p

    @classmethod
    def between(cls, full, reduced) -> "LR":
        from scipy.stats import chi2 as chi2_dist
        df = len(full.params) - len(reduced.params)
        stat = 2 * (full.llf - reduced.llf)
        return cls(stat, df, float(chi2_dist.sf(stat, df)) if df else 1.0)

    def __repr__(self) -> str:
        return f"chi2({self.df})={self.chi2:.1f} p={self.p:.2e}"


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    keep = p.groupby("prop")["forgot"].agg(["size", "sum"])
    keep = keep[(keep["size"] >= MIN_ROWS) & (keep["sum"] >= MIN_EVENTS)].index
    p = p[p["prop"].isin(keep)].reset_index(drop=True)

    for c in ["subj_strain", "obj_strain", "subj_strain_w", "obj_strain_w",
              "evidence_static", "evidence_w", "s_pop", "o_pop",
              "s_rivals_rare", "o_rivals_rare", "num_shared_chunks",
              "s_ntok", "o_ntok", "s_charlen", "o_charlen"]:
        if c in p.columns:
            p["l_" + c] = np.log1p(p[c].astype(float))
    for c in ["s_own_rare", "o_own_rare"]:
        if c in p.columns:
            p["l_" + c] = np.log(p[c].astype(float).clip(lower=1e-6))
    return p


def fit(formula: str, p: pd.DataFrame, cluster: bool = False):
    import statsmodels.formula.api as smf
    kw = dict(disp=0, method="bfgs", maxiter=900)
    if cluster:
        kw.update(cov_type="cluster", cov_kwds={"groups": p["id"].values})
    return smf.logit(formula, data=p).fit(**kw)


def main():
    from scipy.stats import chi2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True)
    a = ap.parse_args()

    panel = pd.read_parquet(a.panel)
    p = prepare(panel).dropna(subset=["ratio_static", "ratio_w",
                                      "l_s_own_rare", "l_o_own_rare"])
    print(f"panel: {len(p)} rows | {int(p['forgot'].sum())} forgetting events | "
          f"{p['id'].nunique()} facts | {p['prop'].nunique()} relations | "
          f"{p['window'].nunique()} windows\n")

    models = MODELS

    fits = {n: fit(f, p) for n, f in models.items()}
    b = fits["exposure only"]
    bl = fits["+ NAME LENGTH"]
    print(f"{'model':<34} {'pseudoR2':>9}  {'LR vs exposure-only':>26}")
    print("-" * 74)
    for n, m in fits.items():
        k = len(m.params) - len(b.params)
        lr = 2 * (m.llf - b.llf)
        tail = "--" if k == 0 else f"chi2({k})={lr:7.1f}  p={chi2.sf(lr, k):.2e}"
        print(f"{n:<34} {m.prsquared:9.4f}  {tail:>26}")

    print("\nIncremental over the NAME-LENGTH baseline (the honest comparison):")
    for n in ["+ strain STATIC | name length", "+ strain WINDOWED | name length",
              "+ F1 ownership/rivals | len", "+ F2 ratio WINDOWED | len",
              "+ everything | name length"]:
        k = len(fits[n].params) - len(bl.params)
        lr = 2 * (fits[n].llf - bl.llf)
        print(f"  {n:<34} chi2({k})={lr:7.1f}  p={chi2.sf(lr, k):.2e}")

    # Does the windowed form add anything the static one does not?
    both = fit(BASE + NAMELEN + STRAIN + STRAIN_W, p)
    for nested, label in [("+ strain STATIC | name length", "windowed on top of static"),
                          ("+ strain WINDOWED | name length", "static on top of windowed")]:
        lr = 2 * (both.llf - fits[nested].llf)
        print(f"\n  {label:<28} chi2(2)={lr:7.1f}  p={chi2.sf(lr, 2):.2e}")

    print("\n" + "=" * 74)
    print("Combined model (name length controlled), SEs clustered by fact:")
    mc = fit(models["+ everything | name length"], p, cluster=True)
    tb = mc.summary2().tables[1]
    show = [c for c in tb.index if not c.startswith("C(") and c != "Intercept"]
    print(tb.loc[show, ["Coef.", "Std.Err.", "P>|z|"]].round(4).to_string())

    print("\nRaw forgetting rate by windowed object-strain quartile:")
    p["q"] = pd.qcut(p["obj_strain_w"].rank(method="first"), 4,
                     labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    print(p.groupby("q", observed=True)["forgot"]
          .agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
