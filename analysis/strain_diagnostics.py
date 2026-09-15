"""Regenerate every diagnostic quoted in ``FINDINGS_token_strain.md``.

Most numbers in that doc originally came from one-off interactive snippets, which
left the prose as the only record.  This script is the reproducible source for all
of them: run it, keep the log next to the findings, and any number in the doc can
be traced back to a command.

Sections mirror the doc:

  1. panel dimensions + relation filter
  2. the reported model: LR ladder and clustered coefficients (with coef x SD)
  3. exposure specification robustness -- why BASE is gold+rival, static only
  4. collinearity structure of the exposure terms
  5. why the windowed exposure term was dropped (the ~46% sampling argument)
  6. mutual suppression between strain and rival exposure
  7. strain vs. distract: they are not the same construct
  8. per-relation breakdown
  9. windowed-vs-static strain
 10. permutation test across exposure specs (optional; ~0.7s/draw/spec)
 11. inherited claims -- the upstream recall sweep, the top-decile cut, entity
     activity per window, object-entity chunk occupancy, and charlen/ntok. These
     came from earlier sessions and were prose-only until now; two of them are
     denominator-sensitive, so the section prints every denominator rather than one.
 12. tokenization claims -- the DF cutoff (which tokens it removes, and the gap
     that makes the choice non-arbitrary) and the case against pooling Wikidata
     aliases. These were stated only in the src/features/strain.py docstring.

Usage:
    python analysis/strain_diagnostics.py \
        --panel data/artifacts/features/strain_panel_15shot_top10.parquet \
        --out data/artifacts/features/strain_diagnostics_specE.txt
    # add --permutations 150 to include section 10 (~2 min per spec)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from analysis.strain_models import (  # noqa: E402
    BASE, MIN_EVENTS, MIN_ROWS, NAMELEN, STRAIN, STRAIN_W, F1, F2, F2_W, LR,
    MODELS, fit, prepare)

# The feature families the report plots, scored against both reference models.
FAMILIES = {
    "strain (static)": STRAIN,
    "strain (windowed)": STRAIN_W,
    "F1 ownership/rivals": F1,
    "F2 alt-answer ratio": F2_W,
}

FE = " + C(prop) + C(window)"
POP = " + l_s_pop + l_o_pop"
# BASE with the rival-exposure term removed, for the suppression ladder.
NO_DISTRACT = "forgot ~ l_shared_static" + POP + FE
EXTRA_LOGS = ["shared_static", "distract_static", "shared_w", "distract_w"]

J: dict = {}          # machine-readable mirror of everything printed


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def head(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def coef_table(m, p, terms=None) -> pd.DataFrame:
    """Clustered coefficients with the coef x SD rescaling the doc sorts by."""
    t = m.summary2().tables[1]
    show = terms or [c for c in t.index
                     if not c.startswith("C(") and c != "Intercept"]
    out = t.loc[show, ["Coef.", "Std.Err.", "P>|z|"]].copy()
    out["SD(x)"] = [p[c].std() for c in show]
    out["coef x SD"] = out["Coef."] * out["SD(x)"]
    out["OR per SD"] = np.exp(out["coef x SD"])
    return out.sort_values("coef x SD", key=abs, ascending=False)


def sec1_panel(raw: pd.DataFrame, p: pd.DataFrame) -> None:
    head(1, "Panel dimensions and the relation filter")
    g = raw.groupby("prop")["forgot"].agg(rows="size", events="sum")
    g = g.sort_values("rows", ascending=False)
    print(f"raw panel: {len(raw)} rows | {int(raw.forgot.sum())} events | "
          f"{raw.id.nunique()} facts | {raw.window.nunique()} windows | "
          f"{raw.prop.nunique()} relations\n")
    print(g.to_string())
    print(f"\nfilter: rows >= {MIN_ROWS} and events >= {MIN_EVENTS}")
    print(f"kept: {len(p)} rows | {int(p.forgot.sum())} events | "
          f"{p.id.nunique()} facts | {p.prop.nunique()} relations")
    f = p.drop_duplicates("id")
    print(f"\nnum_shared_chunks over the {len(f)} modelled facts: "
          f"min {f.num_shared_chunks.min()} | median {f.num_shared_chunks.median():.0f} "
          f"| p90 {f.num_shared_chunks.quantile(.9):.0f} | max {f.num_shared_chunks.max()}")
    J["panel"] = {"raw_rows": len(raw), "raw_events": int(raw.forgot.sum()),
                  "raw_facts": int(raw.id.nunique()), "rows": len(p),
                  "events": int(p.forgot.sum()), "facts": int(p.id.nunique()),
                  "relations": int(p.prop.nunique()),
                  "nsc_median": float(f.num_shared_chunks.median()),
                  "nsc_p90": float(f.num_shared_chunks.quantile(.9))}


def sec2_reported(p: pd.DataFrame) -> None:
    head(2, "The reported model (BASE = gold + rival exposure, static only)")
    print(f"BASE    = {BASE}\nNAMELEN ={NAMELEN}\nSTRAIN  ={STRAIN}\n")
    base, nl = fit(BASE, p), fit(BASE + NAMELEN, p)
    full = fit(BASE + NAMELEN + STRAIN, p)
    print(f"pseudoR2: BASE {base.prsquared:.4f} | +NAMELEN {nl.prsquared:.4f} "
          f"| +STRAIN {full.prsquared:.4f}\n")
    print(f"NAMELEN vs BASE          {LR.between(nl, base)}")
    print(f"STRAIN  vs BASE          {LR.between(full, base)}")
    print(f"STRAIN  vs BASE+NAMELEN  {LR.between(full, nl)}   <-- HEADLINE")
    # Both reference models, side by side. Read the RIGHT column for anything
    # about interference: the left one credits each family with name length's
    # contribution too. (This table is what strain_models.main() used to print.)
    print(f"\n{'model':<34}{'pseudoR2':>10}{'vs BASE':>26}{'vs BASE+NAMELEN':>26}")
    print("-" * 96)
    for n, f in MODELS.items():
        m = fit(f, p)
        a, b = LR.between(m, base), LR.between(m, nl)
        # df <= 0 means the row IS the reference (or is smaller than it), so there
        # is no nested test to report -- not a chi2 of negative degrees of freedom.
        def cell(lr):
            return ("--" if lr.df <= 0
                    else f"chi2({lr.df})={lr.chi2:6.1f} p={lr.p:.1e}")
        print(f"  {n:<32}{m.prsquared:10.4f}{cell(a):>26}{cell(b):>26}")
    print("\nclustered coefficients, sorted by |coef x SD|:")
    ct = coef_table(fit(BASE + NAMELEN + STRAIN, p, cluster=True), p)
    print(ct.round(4).to_string())
    lr = LR.between(full, nl)
    J["reported"] = {"headline_chi2": lr.chi2, "headline_p": lr.p,
                     "pseudoR2": [base.prsquared, nl.prsquared, full.prsquared]}
    # -- persisted for the report, so it never refits anything itself --------
    J["models"] = {}
    for n, f in MODELS.items():
        m = fit(f, p)
        a, b = LR.between(m, base), LR.between(m, nl)
        J["models"][n] = {"pseudoR2": float(m.prsquared),
                          "vs_base": [a.chi2, a.df, a.p],
                          "vs_namelen": [b.chi2, b.df, b.p]}
    J["families"] = {}
    for n, terms in FAMILIES.items():
        m = fit(BASE + NAMELEN + terms, p)
        a, b = LR.between(m, base), LR.between(m, nl)
        J["families"][n] = {"vs_base": [a.chi2, a.df, a.p],
                            "vs_namelen": [b.chi2, b.df, b.p]}
    nlr = LR.between(nl, base)
    J["namelen_lr"] = [nlr.chi2, nlr.df, nlr.p]
    J["coefs"] = [{"term": t, "coef": float(r["Coef."]), "se": float(r["Std.Err."]),
                   "sd": float(r["SD(x)"]), "std": float(r["coef x SD"]),
                   "std_se": float(r["Std.Err."] * r["SD(x)"]),
                   "p": float(r["P>|z|"])}
                  for t, r in ct.iterrows()]


def sec3_exposure_specs(p: pd.DataFrame) -> None:
    head(3, "Exposure specification robustness (why BASE is what it is)")
    specs = {
        "A pooled, static+windowed (old BASE)": "l_evidence_w + l_evidence_static",
        "B gold only, static":                  "l_shared_static",
        "C gold only, static+windowed":         "l_shared_w + l_shared_static",
        "D decomposed, static+windowed":        "l_shared_w + l_shared_static "
                                                "+ l_distract_w + l_distract_static",
        "E decomposed, static only (CURRENT)":  "l_shared_static + l_distract_static",
        "G pooled, static only":                "l_evidence_static",
    }
    # The "current" row must reproduce BASE exactly, or this whole comparison is
    # measuring a stale specification. Fail loudly rather than quietly.
    rebuilt = "forgot ~ " + specs["E decomposed, static only (CURRENT)"] + POP + FE
    if rebuilt != BASE:
        raise SystemExit(
            "section 3's 'CURRENT' spec no longer reproduces BASE.\n"
            f"  rebuilt: {rebuilt}\n  BASE   : {BASE}\n"
            "Update the specs dict (or POP/FE) to match strain_models.BASE."
        )
    print(f"{'exposure control':<38}{'strain chi2':>12}{'p':>10}"
          f"{'subj coef':>11}{'p':>9}{'exp terms':>11}{'unident.':>10}")
    print("-" * 101)
    rows = {}
    for n, ex in specs.items():
        b = "forgot ~ " + ex + POP + FE
        lr = LR.between(fit(b + NAMELEN + STRAIN, p), fit(b + NAMELEN, p))
        t = fit(b + NAMELEN + STRAIN, p, cluster=True).summary2().tables[1]
        ex_t = [x for x in t.index
                if any(k in x for k in ("shared", "distract", "evidence"))]
        bad = sum(1 for x in ex_t
                  if abs(t.loc[x, "Coef."] / t.loc[x, "Std.Err."]) < 1)
        print(f"{n:<38}{lr.chi2:12.1f}{lr.p:10.1e}"
              f"{t.loc['l_subj_strain','Coef.']:+11.4f}"
              f"{t.loc['l_subj_strain','P>|z|']:9.4f}{len(ex_t):11d}{bad:10d}")
        rows[n] = {"chi2": lr.chi2, "p": lr.p,
                   "subj_coef": float(t.loc["l_subj_strain", "Coef."]),
                   "subj_p": float(t.loc["l_subj_strain", "P>|z|"]),
                   "exposure_terms": len(ex_t), "unidentified": bad}
    print("\n'unident.' counts exposure terms with |coef/SE| < 1.  Note D is the worst")
    print("option: decomposing while KEEPING the windowed forms gives 4 terms in 2")
    print("near-collinear pairs.  Dropping the windowed term is what buys identification.")
    J["exposure_specs"] = rows


def sec4_collinearity(raw: pd.DataFrame) -> None:
    head(4, "Collinearity structure of the exposure terms")
    c = ["shared_static", "shared_w", "distract_static", "distract_w",
         "evidence_static", "evidence_w"]
    L = pd.DataFrame({x: np.log1p(raw[x].astype(float)) for x in c})
    m = L.corr(method="spearman")
    print("spearman, log1p scale:\n")
    print(m.round(3).to_string())
    print("\nThe damaging pairs are static-vs-windowed (~0.999), NOT gold-vs-rival")
    print("(0.736).  That is why splitting gold/rival does not by itself fix the")
    print("standard errors, and dropping the windowed term does.")
    J["collinearity"] = m.round(4).to_dict()


def sec5_why_no_windowed(raw: pd.DataFrame) -> None:
    head(5, "Why the windowed exposure term was dropped (~46% uniform sampling)")
    r = (raw.evidence_w / raw.evidence_static.replace(0, np.nan)).dropna()
    print(f"evidence_w / evidence_static per row: mean {r.mean():.3f} "
          f"sd {r.std():.3f} | p5 {r.quantile(.05):.3f} median {r.median():.3f} "
          f"p95 {r.quantile(.95):.3f}")
    print(f"expected for a 50,000-step window: 50000/109672 = {50000/109672:.3f}\n")
    g = (raw.assign(ratio=raw.evidence_w / raw.evidence_static.replace(0, np.nan))
            .groupby("window").ratio.agg(["mean", "std", "size"]))
    g["window_len"] = raw.groupby("window").apply(
        lambda d: int(d.step_hi.iloc[0] - d.step_lo.iloc[0]))
    g["len/epoch"] = g.window_len / 109672
    print("per-window sampling fraction vs window length:")
    print(g.round(3).to_string())
    print("\nThe measured fraction tracks window length (the final window is 58,032")
    print("steps and sits at its own 0.529), which independently confirms the")
    print("WindowMap epoch arithmetic.")
    x = np.log1p(raw.evidence_static.astype(float))
    y = np.log1p(raw.evidence_w.astype(float))
    b = np.polyfit(x, y, 1)
    resid = y - np.polyval(b, x)
    r2 = 1 - resid.var() / y.var()
    print(f"\nlog-log fit: l_evidence_w = {b[0]:.3f} * l_evidence_static {b[1]:+.3f}")
    print(f"  R2 = {r2:.4f} | residual sd {resid.std():.3f} vs outcome sd {y.std():.3f}")
    print("  => slope 1, intercept -ln(0.46): a constant offset C(window) absorbs.")
    J["windowed_exposure"] = {"ratio_mean": float(r.mean()), "ratio_sd": float(r.std()),
                              "expected": 50000/109672, "slope": float(b[0]),
                              "intercept": float(b[1]), "r2": float(r2),
                              "resid_sd": float(resid.std())}


def sec6_suppression(p: pd.DataFrame) -> None:
    head(6, "Mutual suppression between strain and rival exposure")
    ladder = {
        "1 distract alone + FE":             "forgot ~ l_distract_static" + FE,
        "2 + gold exposure":                 "forgot ~ l_distract_static + l_shared_static" + FE,
        "3 + popularity  (= BASE)":          BASE,
        "4 + name length (= BASE+NAMELEN)":  BASE + NAMELEN,
        "5 + strain      (= reported)":      BASE + NAMELEN + STRAIN,
        "6 as 5, gold exposure dropped":     "forgot ~ l_distract_static" + POP + FE + NAMELEN + STRAIN,
    }
    print(f"{'model':<36}{'l_distract_static':>20}{'l_subj_strain':>20}")
    print("-" * 76)
    rows = {}
    for n, f in ladder.items():
        t = fit(f, p, cluster=True).summary2().tables[1]

        def g(k):
            if k not in t.index:
                return "--"
            return f"{t.loc[k,'Coef.']:+.4f} (p={t.loc[k,'P>|z|']:.3f})"
        print(f"{n:<36}{g('l_distract_static'):>20}{g('l_subj_strain'):>20}")
        rows[n] = {k: (None if k not in t.index else
                       [float(t.loc[k, "Coef."]), float(t.loc[k, "P>|z|"])])
                   for k in ("l_distract_static", "l_subj_strain")}
    print("\nSign never reverses; significance is not stable.  Increments:")
    d_no = LR.between(fit(BASE + NAMELEN, p), fit(NO_DISTRACT + NAMELEN, p))
    d_yes = LR.between(fit(BASE + NAMELEN + STRAIN, p),
                       fit(NO_DISTRACT + NAMELEN + STRAIN, p))
    s_yes = LR.between(fit(BASE + NAMELEN + STRAIN, p), fit(BASE + NAMELEN, p))
    s_no = LR.between(fit(NO_DISTRACT + NAMELEN + STRAIN, p),
                      fit(NO_DISTRACT + NAMELEN, p))
    print(f"  distract, without strain   {d_no}")
    print(f"  distract, with strain      {d_yes}")
    print(f"  strain, without distract   {s_no}")
    print(f"  strain, with distract      {s_yes}")
    t = fit(NO_DISTRACT + NAMELEN + STRAIN, p, cluster=True).summary2().tables[1]
    print(f"\n  strain coef without distract: "
          f"{t.loc['l_subj_strain','Coef.']:+.4f} (p={t.loc['l_subj_strain','P>|z|']:.4f})")
    print("  Each roughly doubles the other's increment -- two negatively correlated")
    print("  predictors with opposite-signed effects.  Read DIRECTIONS as robust and")
    print("  MAGNITUDES as jointly determined.  Row 3 is the diagnostic for the")
    print("  'prominence not competition' reading: popularity, not name length, is")
    print("  what removes distract's significance.")
    J["suppression"] = {"ladder": rows,
                        "distract_no_strain": [d_no.chi2, d_no.p],
                        "distract_with_strain": [d_yes.chi2, d_yes.p],
                        "strain_no_distract": [s_no.chi2, s_no.p],
                        "strain_with_distract": [s_yes.chi2, s_yes.p]}


def sec7_strain_vs_distract(raw: pd.DataFrame) -> None:
    head(7, "Strain and distract are not the same construct")
    c = ["subj_strain", "obj_strain", "distract_static", "shared_static",
         "s_ntok", "o_ntok"]
    L = pd.DataFrame({x: np.log1p(raw[x].astype(float)) for x in c})
    print("spearman, log1p scale:\n")
    print(L.corr(method="spearman").round(3).to_string())
    print(f"\nunits differ by three orders of magnitude:")
    print(f"  distract_static  median {raw.distract_static.median():8.0f}  "
          f"max {raw.distract_static.max():8d}   (CHUNKS)")
    print(f"  subj_strain      median {raw.subj_strain.median():8.0f}  "
          f"max {raw.subj_strain.max():8.0f}   (ENTITIES)")
    print("\nsubj_strain ~ distract_static is near zero and slightly NEGATIVE, while")
    print("subj_strain ~ s_ntok is strong and distract_static ~ shared_static is")
    print("strong: they belong to different confound families (name length vs")
    print("exposure).  strain is corpus-independent -- computed from name strings and")
    print("the entity universe, never touching chunk data -- while distract is")
    print("nothing but a count of corpus co-occurrences.")
    J["strain_vs_distract"] = {
        "spearman_strain_distract": float(
            L.subj_strain.corr(L.distract_static, method="spearman")),
        "spearman_strain_ntok": float(
            L.subj_strain.corr(L.s_ntok, method="spearman")),
        "spearman_distract_shared": float(
            L.distract_static.corr(L.shared_static, method="spearman"))}


def sec8_per_relation(p: pd.DataFrame) -> None:
    head(8, "Per-relation subject-strain coefficient")
    t = fit(BASE + NAMELEN + STRAIN, p, cluster=True).summary2().tables[1]
    print(f"{'relation':<14}{'rows':>6}{'events':>8}   l_subj_strain")
    print(f"{'all':<14}{len(p):6d}{int(p.forgot.sum()):8d}   "
          f"{t.loc['l_subj_strain','Coef.']:+.3f} "
          f"(se {t.loc['l_subj_strain','Std.Err.']:.3f}, "
          f"p {t.loc['l_subj_strain','P>|z|']:.3f})")
    rows = {}
    full_rows = [{"relation": "all", "rows": len(p),
                  "events": int(p.forgot.sum()),
                  "coef": float(t.loc["l_subj_strain", "Coef."]),
                  "se": float(t.loc["l_subj_strain", "Std.Err."]),
                  "p": float(t.loc["l_subj_strain", "P>|z|"])}]
    # within one relation the C(prop) term is constant, so drop it
    f_rel = ("forgot ~ l_shared_static + l_distract_static" + POP
             + " + C(window)" + NAMELEN + STRAIN)
    for r, g in p.groupby("prop"):
        try:
            tt = fit(f_rel, g, cluster=True).summary2().tables[1]
            print(f"{r:<14}{len(g):6d}{int(g.forgot.sum()):8d}   "
                  f"{tt.loc['l_subj_strain','Coef.']:+.3f} "
                  f"(se {tt.loc['l_subj_strain','Std.Err.']:.3f}, "
                  f"p {tt.loc['l_subj_strain','P>|z|']:.3f})")
            rows[r] = [float(tt.loc["l_subj_strain", "Coef."]),
                       float(tt.loc["l_subj_strain", "P>|z|"])]
            full_rows.append({"relation": r, "rows": len(g),
                              "events": int(g.forgot.sum()),
                              "coef": float(tt.loc["l_subj_strain", "Coef."]),
                              "se": float(tt.loc["l_subj_strain", "Std.Err."]),
                              "p": float(tt.loc["l_subj_strain", "P>|z|"])})
        except Exception as e:                      # defensive: fit failed to converge
            print(f"{r:<14}{len(g):6d}{int(g.forgot.sum()):8d}   FAILED {e!s:.40}")
    J["per_relation"] = rows
    J["per_relation_full"] = full_rows


def sec9_windowed_strain(p: pd.DataFrame) -> None:
    head(9, "Windowed vs static strain")
    both = fit(BASE + NAMELEN + STRAIN + STRAIN_W, p)
    stat = fit(BASE + NAMELEN + STRAIN, p)
    wind = fit(BASE + NAMELEN + STRAIN_W, p)
    a, b = LR.between(both, stat), LR.between(both, wind)
    print(f"windowed on top of static   {a}")
    print(f"static on top of windowed   {b}")
    print("\nNeither adds to the other: ~90% of the 19,384-entity universe is active")
    print("in every 50,000-step window, so windowed strain is a near-copy of static.")
    print("NOTE this is SATURATION -- a different mechanism from the uniform-sampling")
    print("argument that retired the windowed EXPOSURE term (section 5).")
    J["windowed_strain"] = {"windowed_on_static": [a.chi2, a.p],
                            "static_on_windowed": [b.chi2, b.p]}


def sec10_permutations(p: pd.DataFrame, n: int) -> None:
    head(10, f"Permutation test across exposure specs ({n} draws each)")
    # The loop itself lives in null_control_distribution.run_permutations, so the
    # standalone script and this section cannot drift apart.
    from analysis.null_control_distribution import run_permutations
    from feature_extraction.build_strain_panel import load_popqa
    from src.features import strain as S
    df = load_popqa()
    meta = df.set_index("id")
    u = S.EntityUniverse.from_popqa(df)
    real = S.TokenIndex(u, u.tokens(), df_cutoff=S.DF_CUTOFF)
    # The reported spec is BASE + NAMELEN verbatim -- not a hand-rebuilt copy of it.
    specs = {
        "E decomposed static only (CUR)": BASE + NAMELEN,
        "A pooled static+windowed (old)": ("forgot ~ l_evidence_w + l_evidence_static"
                                           + POP + FE + NAMELEN),
        "G pooled static only":           ("forgot ~ l_evidence_static"
                                           + POP + FE + NAMELEN),
    }
    rows = {}
    for nm, F in specs.items():
        r = run_permutations(p, meta, u, real, F, n, verbose=False)
        null = np.asarray(r["null"])
        obs, pe = r["observed"], r["p_empirical"]
        ge = int((null >= obs).sum())
        print(f"{nm:<34} obs={obs:6.2f}  null med={np.median(null):5.2f} "
              f"mean={null.mean():5.2f}  {ge:3d}/{n}  p={pe:.3f}")
        rows[nm] = {"observed": float(obs), "ge": ge, "n": n, "p": pe,
                    "null_median": float(np.median(null)),
                    "null_mean": float(null.mean())}
        if nm.endswith("(CUR)"):
            # This is the reported test; persist the full draw list so the report's
            # histogram can be rebuilt without rerunning the permutations.
            J["permutation_reported"] = r
    print("\nThe permutation p is flat across exposure specifications, so the")
    print("respecification cannot be mistaken for spec-shopping -- and it does not")
    print("rescue significance either.  Unlike the LR chi2 values elsewhere in this")
    print("log, this test is robust to the unclustered-likelihood issue, because the")
    print("real and scrambled statistics are computed identically.")
    J["permutations"] = rows


def sec11_inherited(panel_path: str, recall_dir: str, broad_panel: str) -> None:
    head(11, "Inherited claims (upstream sweep, corpus occupancy, tokenization)")
    import glob
    import os
    from feature_extraction.build_strain_panel import load_popqa
    from src.features import strain as S

    rec = {}
    # -- the recall sweep the panel is built from ------------------------------
    files = sorted(glob.glob(os.path.join(recall_dir, "recall_step*.parquet")))
    files = [f for f in files if ".partial." not in os.path.basename(f)]
    if not files:
        print(f"[skip] no recall parquets under {recall_dir}")
    else:
        rows = []
        for f in files:
            d = pd.read_parquet(f, columns=["id", "step", "correct", "excluded"])
            keep = d[~d["excluded"].astype(bool)]
            rows.append((int(d.step.iloc[0]), len(d), int(d.excluded.sum()),
                         d.correct.mean(), keep.correct.mean()))
        t = pd.DataFrame(rows, columns=["step", "n_facts", "n_excluded",
                                        "acc_all", "acc_kept"]).sort_values("step")
        print("recall sweep (accuracy with `excluded` facts dropped):")
        print(t.round(4).to_string(index=False))
        print(f"\n  {len(t)} checkpoints | {t.n_facts.iloc[0]} facts | "
              f"{t.n_excluded.iloc[0]} excluded | {t.n_facts.iloc[0]-t.n_excluded.iloc[0]} kept")
        print(f"  accuracy {t.acc_kept.iloc[0]:.3f} -> {t.acc_kept.iloc[-1]:.3f} "
              f"(non-monotonic; dips at 200-300K, which is where the events are)")
        rec["sweep"] = {"checkpoints": len(t), "facts": int(t.n_facts.iloc[0]),
                        "excluded": int(t.n_excluded.iloc[0]),
                        "acc_first": float(t.acc_kept.iloc[0]),
                        "acc_last": float(t.acc_kept.iloc[-1])}

    df = load_popqa()
    u = S.EntityUniverse.from_popqa(df)

    # -- the top-decile cut ---------------------------------------------------
    n42 = int((df.num_shared_chunks >= 42).sum())
    print(f"\ntop-decile cut: popqa {len(df)} facts | num_shared_chunks >= 42 -> "
          f"{n42} = {100*n42/len(df):.2f}%")
    print("  (the evaluated 1,452 is this minus the 16 demonstration subjects)")
    rec["top_decile"] = {"popqa_facts": len(df), "n_ge_42": n42,
                         "pct": 100 * n42 / len(df)}

    # -- entity activity per window: the "~90%" claim -------------------------
    try:
        from feature_extraction.build_strain_panel import epoch_paths
        steps = sorted(int(x) for x in
                       pd.read_parquet(panel_path, columns=["step_lo"]).step_lo.unique())
        steps = steps + [658032]
        wm = S.WindowMap.from_batch_indices(epoch_paths(), steps)
        act = np.array([len(wm.active_entities(u, w)) for w in range(wm.n_windows)])
        N = len(u)
        print(f"\nentity activity per window (universe {N}):")
        print(f"  counts {list(act)}")
        print(f"  fraction: min {act.min()/N:.3f} | mean {act.mean()/N:.3f} "
              f"| max {act.max()/N:.3f}   <- the '~90% active' claim")
        print("  This is why windowed STRAIN saturates: the competitor set barely shrinks.")
        rec["activity"] = {"universe": N, "min": float(act.min()/N),
                           "mean": float(act.mean()/N), "max": float(act.max()/N)}
    except FileNotFoundError as e:
        print(f"\n[skip] entity activity needs all six batch_indices files: {e}")

    # -- object-entity chunk occupancy: DENOMINATOR-SENSITIVE -----------------
    mx = max(int(c.max()) for c in u.chunks.values() if c.size)
    uo = np.array([u.chunks[q].size for q in df.o_uri.unique() if q in u.chunks])
    po = np.array([u.chunks[q].size for q in df.o_uri if q in u.chunks])
    allq = np.array([c.size for c in u.chunks.values()])
    ids = pd.read_parquet(panel_path, columns=["id"]).id.unique()
    sub = df[df.id.isin(ids)]
    so = np.array([u.chunks[q].size for q in sub.o_uri.unique() if q in u.chunks])
    print(f"\nobject-entity chunk occupancy (corpus max chunk id {mx} ~ {mx/1e6:.1f}M):")
    print(f"  unique object entities   ({len(uo):5d}) median {np.median(uo):7.0f}")
    print(f"  per-FACT object rows     ({len(po):5d}) median {np.median(po):7.0f}"
          "   <- the doc's '305'")
    print(f"  all entities             ({len(allq):5d}) median {np.median(allq):7.0f}")
    print(f"  objects in THIS panel    ({len(so):5d}) median {np.median(so):7.0f}"
          "   <- the relevant one")
    # An entity with k chunks is rarely absent from a window covering fraction f
    # unless f <~ 1/k; f = steps/109672.
    for lbl, k in (("per-fact median", np.median(po)),
                   ("this panel's median", np.median(so))):
        print(f"  window size for absence to be common at {lbl} k={k:.0f}: "
              f"<~ {109672/k:.0f} steps")
    print("  NOTE the doc's 'roughly 1-2K steps' follows from the per-fact 305, but")
    print("  the facts actually analysed are the top decile, whose objects are far")
    print("  more frequent -- so the requirement is much stricter than stated.")
    rec["occupancy"] = {"corpus_max_chunk": mx,
                        "median_unique_object": float(np.median(uo)),
                        "median_per_fact_object": float(np.median(po)),
                        "median_all_entities": float(np.median(allq)),
                        "median_panel_object": float(np.median(so)),
                        "steps_per_fact": float(109672/np.median(po)),
                        "steps_panel": float(109672/np.median(so))}

    # -- charlen vs ntok: PANEL-SENSITIVE ------------------------------------
    print("\ncharlen ~ ntok spearman, per panel:")
    rec["charlen"] = {}
    for nm, f in (("broad strain_panel", broad_panel), ("this panel", panel_path)):
        if not Path(f).exists():
            print(f"  [skip] {f} not present")
            continue
        q = pd.read_parquet(f)
        sp = (pd.Series(q.s_ntok).corr(q.s_charlen, method="spearman"),
              pd.Series(q.o_ntok).corr(q.o_charlen, method="spearman"))
        print(f"  {nm:<20} rows {len(q):6d}  subj {sp[0]:.3f}  obj {sp[1]:.3f}")
        rec["charlen"][nm] = [float(sp[0]), float(sp[1])]
    print("  The doc's '0.88 / 0.84' is the BROAD panel; this panel is lower.")
    J["inherited"] = rec


def sec12_tokenization(panel_path: str) -> None:
    head(12, "Tokenization claims (DF cutoff, canonical forms vs Wikidata aliases)")
    from feature_extraction.build_strain_panel import POPQA_COLS, load_popqa
    from src.features import strain as S

    df = load_popqa(POPQA_COLS + ["s_aliases", "o_aliases"])
    u = S.EntityUniverse.from_popqa(df)
    rec: dict = {}

    # -- the DF cutoff -------------------------------------------------------
    dfreq = S.token_document_frequency(u.tokens())
    print(f"universe {len(u)} entities | {len(dfreq)} distinct word tokens | "
          f"DF_CUTOFF = {S.DF_CUTOFF}\n")
    print("top 25 tokens by number of entities whose canonical name contains them:")
    top = dfreq.most_common(25)
    for i, (t, c) in enumerate(top, 1):
        flag = "  <- removed" if c > S.DF_CUTOFF else ""
        print(f"  {i:2d}  {t:<10}{c:6d}{flag}")
    removed = sorted(t for t, c in dfreq.items() if c > S.DF_CUTOFF)
    kept_max = max(c for c in dfreq.values() if c <= S.DF_CUTOFF)
    rem_min = min(c for c in dfreq.values() if c > S.DF_CUTOFF)
    print(f"\nremoved by the cutoff: {removed}")
    print(f"gap: largest kept DF = {kept_max}, smallest removed DF = {rem_min}")
    print(f"  => ANY cutoff in [{kept_max}, {rem_min}) removes exactly this set, so "
          f"{S.DF_CUTOFF} is a point in a gap, not a tuned value.")
    print("  Below the gap the distribution is a smooth Zipf with function words and")
    print("  collision-bearing names interleaved -- a lower cutoff removes both:")
    names = ["john", "david", "county", "and", "in"]
    print("   " + " | ".join(f"{w}={dfreq[w]}" for w in names))
    rec["df_cutoff"] = {"cutoff": S.DF_CUTOFF, "removed": removed,
                        "gap": [kept_max, rem_min],
                        "top25": [[t, c] for t, c in top],
                        "named": {w: dfreq[w] for w in names}}

    # -- canonical forms vs aliases -------------------------------------------
    forms: dict = {}
    aliases: dict = {}
    for q, t, a in (list(zip(df.s_uri, df.subj, df.s_aliases))
                    + list(zip(df.o_uri, df.obj, df.o_aliases))):
        forms.setdefault(q, set()).add(str(t))
        bag = aliases.setdefault(q, set())
        if a is not None:
            bag.update(str(x) for x in a)
    multi = sum(1 for v in forms.values() if len(v) > 1)
    all_forms = {q: forms[q] | aliases[q] for q in forms}
    n_can = sum(len(v) for v in forms.values())
    n_all = sum(len(v) for v in all_forms.values())
    max_all = max(len(v) for v in all_forms.values())
    print(f"\nobserved surface forms: {len(forms)} QIDs, {multi} with more than one "
          "(so canonical form = the one form popqa-kas actually uses)")
    print(f"adding Wikidata aliases: {n_can} -> {n_all} forms = {n_all/n_can:.2f}x; "
          f"most for one entity: {max_all} forms")

    can = S.TokenIndex(u, u.tokens(), df_cutoff=S.DF_CUTOFF)
    pooled_tok = {q: set().union(*(S.word_tokens(f) for f in all_forms[q]))
                  for q in forms}
    ids = set(pd.read_parquet(panel_path, columns=["id"]).id)
    sub = df[df.id.isin(ids)]
    print("\nmedian OBJECT strain, canonical forms vs alias-pooled. The effect is")
    print("denominator- and cutoff-sensitive, so every combination is printed:\n")
    print(f"  {'cutoff':<8}{'denominator':<26}{'canonical':>10}{'pooled':>9}{'x':>7}")
    grid = []
    for cn, cut in (("400", S.DF_CUTOFF), ("none", None)):
        ci = S.TokenIndex(u, u.tokens(), df_cutoff=cut)
        pi = S.TokenIndex(u, pooled_tok, df_cutoff=cut)
        for dn, frame in (("per fact (all 14,267)", df), ("per fact (this panel)", sub)):
            a = np.median([ci.strain(o, exclude={s_}) for s_, o in zip(frame.s_uri, frame.o_uri)])
            b = np.median([pi.strain(o, exclude={s_}) for s_, o in zip(frame.s_uri, frame.o_uri)])
            print(f"  {cn:<8}{dn:<26}{a:10.0f}{b:9.0f}{b/max(a,1):7.1f}")
            grid.append({"cutoff": cn, "denominator": dn,
                         "canonical": float(a), "pooled": float(b)})
    print("\n  Pooling inflates median object strain several-fold under every")
    print("  definition. NOTE: the strain.py docstring's '5 to 251' does not reproduce")
    print("  under any of these (nor under per-unique-entity medians); the direction")
    print("  is robust, that specific pair of numbers is not.")
    rec["aliases"] = {"qids": len(forms), "qids_multi_form": multi,
                      "forms_canonical": n_can, "forms_with_aliases": n_all,
                      "max_forms_one_entity": max_all, "strain_grid": grid}
    _ = can  # canonical index at the reported cutoff (grid row 1)
    J["tokenization"] = rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel",
                    default="data/artifacts/features/strain_panel_15shot_top10.parquet")
    ap.add_argument("--out",
                    default="data/artifacts/features/strain_diagnostics_specE.txt")
    ap.add_argument("--json", default=None,
                    help="also write the machine-readable mirror (default: --out with .json)")
    ap.add_argument("--permutations", type=int, default=0,
                    help="draws per exposure spec for section 10 (0 = skip; ~0.7s each)")
    ap.add_argument("--recall-dir",
                    default="data/artifacts/expermints/"
                            "15shot_trajectory_top10_internal_with_hidden/recall",
                    help="recall sweep the panel was built from (section 11)")
    ap.add_argument("--broad-panel",
                    default="data/artifacts/features/strain_panel.parquet",
                    help="the wider panel, for the charlen comparison (section 11)")
    ap.add_argument("--skip-inherited", action="store_true",
                    help="skip sections 11-12 (they reload popqa and the batch "
                         "indices, ~45s)")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, fh)
    try:
        import statsmodels
        print("strain diagnostics -- regenerates every number in "
              "FINDINGS_token_strain.md")
        print(f"panel   : {a.panel}")
        print(f"pandas {pd.__version__} | numpy {np.__version__} | "
              f"statsmodels {statsmodels.__version__}")
        raw = pd.read_parquet(a.panel)
        p = prepare(raw)
        for c in EXTRA_LOGS:                    # harmless if prepare() added them
            if "l_" + c not in p.columns:
                p["l_" + c] = np.log1p(p[c].astype(float))
        sec1_panel(raw, p)
        sec2_reported(p)
        sec3_exposure_specs(p)
        sec4_collinearity(raw)
        sec5_why_no_windowed(raw)
        sec6_suppression(p)
        sec7_strain_vs_distract(raw)
        sec8_per_relation(p)
        sec9_windowed_strain(p)
        if a.permutations:
            sec10_permutations(p, a.permutations)
        else:
            head(10, "Permutation test -- SKIPPED (pass --permutations 150)")
        if a.skip_inherited:
            head(11, "Inherited claims -- SKIPPED (--skip-inherited)")
            head(12, "Tokenization claims -- SKIPPED (--skip-inherited)")
        else:
            sec11_inherited(a.panel, a.recall_dir, a.broad_panel)
            sec12_tokenization(a.panel)
    finally:
        sys.stdout = real_stdout
        fh.close()

    jp = Path(a.json) if a.json else out.with_suffix(".json")
    jp.write_text(json.dumps(J, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}\nwrote {jp}")


if __name__ == "__main__":
    main()
