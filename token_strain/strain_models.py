"""Model specification for the forgetting analysis. **Library only -- no CLI.**

This module is the single source of truth for *what* is fitted: the formulas
(`BASE`, `NAMELEN`, `STRAIN`, ...), the relation filter (`MIN_ROWS`/`MIN_EVENTS`),
the panel preparation (`prepare`), the fitter (`fit`) and the likelihood-ratio test
(`LR`). It deliberately does **not** fit anything at import time and has no
`main()`.

There is exactly one entry point that fits these models and reports them:

    python token_strain/strain_diagnostics.py          # every number, saved to a log

`token_strain/null_control_distribution.py` also fits, but only the permutation
replicates, and it exposes that loop as `run_permutations` so the diagnostics uses
the same implementation rather than a copy. `token_strain/report_token_strain.py` fits
nothing at all -- it renders the diagnostics JSON.

It used to have its own `main()` printing a model table. That meant two scripts
fitting the same models and two logs to keep in sync, so it was folded into the
diagnostics; `strain_diagnostics.py` section 2 prints the same ladder against both
reference models.

The question the specification answers: does token interference explain forgetting
*on top of* how much the model saw the fact? So every feature family is scored as an
addition to an exposure baseline, never against an empty model -- `distract`,
`strain` and `evidence` are all correlated with plain frequency, and without that
control the interference features mostly re-measure exposure.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

# Relations with too few rows or forgetting events to estimate their own intercept
# with any precision are dropped. This is a precision judgment, NOT a numerical
# necessity: keeping them neither makes the design matrix singular nor causes
# separation -- the fit on all relations converges -- so the filter is a choice about
# what a relation-level intercept can bear, not a fix for a failing fit.
MIN_ROWS, MIN_EVENTS = 100, 20

# --------------------------------------------------------------------------- #
# Model specification -- the single source of truth.  `null_control_distribution`
# and the plotting script both import these, so a change to the baseline can
# never leave one of them fitting a different model than the others.
# --------------------------------------------------------------------------- #
# Exposure control: total co-occurrence of the subject with a candidate answer,
# split into the GOLD answer (`shared_static`, identical to `num_shared_chunks`)
# and RIVAL answers of the same relation (`distract_static`).  The two are only
# correlated at spearman 0.74, and they turn out to act in different ways, so
# pooling them into `evidence` hid a real distinction.
#
# NO WINDOWED EXPOSURE TERM, deliberately.  A 50K-step window samples a constant
# ~46% of EVERY fact's chunks (50000/109672 = 0.456; measured 0.453 +- 0.039, and
# the longer final window sits at 0.529 = 58032/109672).  The fraction is the same
# for every fact, so in log space the windowed count is just the static one minus a
# constant -- `l_evidence_w = 0.997*l_evidence_static - 0.771`, R2 = 0.998 -- and
# `C(window)` already absorbs that offset.  Carrying both left the two exposure
# coefficients unidentified (SEs ~4x the coefficients, p=0.59/0.79) while adding
# nothing: dropping the windowed term moves the strain LR chi2 by 0.0.  This is a
# sampling-uniformity argument and NOT the reason windowed *strain* is degenerate
# -- that one is genuine saturation (~90% of entities active in every window).
BASE = ("forgot ~ l_shared_static + l_distract_static + l_s_pop + l_o_pop "
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
              "shared_static", "distract_static", "shared_w", "distract_w",
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
