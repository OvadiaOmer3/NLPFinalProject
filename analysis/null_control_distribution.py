"""Repeat the token-index permutation N times to get a proper null distribution.

The report shows a single randomised index (seed 0). One draw can be lucky, so
this re-runs the permutation many times and reports where the *real* index's
likelihood-ratio statistic falls inside the resulting null distribution.

Reading the output:

  * If the real chi2 sits in the upper tail of the permuted ones, the feature is
    carrying signal that depends on token identity -- the hypothesis survives.
  * If it sits in the middle (or below), the feature is indistinguishable from
    its own scrambled version, and whatever it measures is a property the
    permutation preserved -- here, how many tokens an entity's name has.

    python analysis/null_control_distribution.py --n 20
"""
from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from analysis.strain_models import BASE, LR, NAMELEN  # noqa: E402
from analysis.strain_models import MIN_EVENTS, MIN_ROWS, fit  # noqa: E402
from feature_extraction.build_strain_panel import load_popqa  # noqa: E402
from src.features import strain as S  # noqa: E402


def permuted_index(universe, real, seed: int) -> S.TokenIndex:
    """Deal each entity a fresh set of tokens of the same size.

    Preserves per-entity token count and the global multiset of tokens; destroys
    which token belongs to whom.  Identical to the report's version except for
    the seed.
    """
    rng = random.Random(seed)
    # sorted(), not set order: Python randomises string hashes per process, so
    # without this the pool order -- and therefore the whole permutation --
    # differs between runs even at a fixed seed.
    kept = {q: sorted(t for t in ts if real.df_full[t] <= S.DF_CUTOFF)
            for q, ts in sorted(universe.tokens().items())}
    pool = [t for ts in kept.values() for t in ts]
    rng.shuffle(pool)
    fake, i = {}, 0
    for q, ts in kept.items():
        fake[q] = set(pool[i:i + len(ts)])
        i += len(ts)
    return S.TokenIndex(universe, fake, df_cutoff=None)


def strain_columns(idx: S.TokenIndex, ids, meta) -> tuple[np.ndarray, np.ndarray]:
    """log1p subject- and object-side strain for each panel row."""
    cache: dict = {}

    def side(qid, other):
        key = (qid, other)
        if key not in cache:
            cache[key] = idx.strain(qid, exclude={other})
        return cache[key]

    subj, obj = [], []
    for fid in ids:
        r = meta.loc[fid]
        subj.append(side(r["s_uri"], r["o_uri"]))
        obj.append(side(r["o_uri"], r["s_uri"]))
    return np.log1p(subj), np.log1p(obj)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="data/artifacts/features/strain_panel.parquet")
    ap.add_argument("--n", type=int, default=20, help="permutations to draw")
    ap.add_argument("--out", default="data/artifacts/features/null_control.json",
                    help="where to cache the distribution for the report")
    ap.add_argument("--control-namelen", action="store_true",
                    help="put name length in the baseline, so the test asks whether "
                         "token identity carries anything BEYOND how long the names "
                         "are -- the sharper question, since strain is largely a "
                         "count of a name's tokens")
    args = ap.parse_args()

    raw = pd.read_parquet(args.panel)
    df = load_popqa()
    meta = df.set_index("id")
    universe = S.EntityUniverse.from_popqa(df)
    real = S.TokenIndex(universe, universe.tokens(), df_cutoff=S.DF_CUTOFF)

    keep = raw.groupby("prop")["forgot"].agg(["size", "sum"])
    keep = keep[(keep["size"] >= MIN_ROWS) & (keep["sum"] >= MIN_EVENTS)].index
    panel = raw[raw["prop"].isin(keep)].reset_index(drop=True)
    cols = ["evidence_w", "evidence_static", "s_pop", "o_pop"]
    if args.control_namelen:
        cols += ["s_ntok", "o_ntok", "s_charlen", "o_charlen"]
    for c in cols:
        panel["l_" + c] = np.log1p(panel[c].astype(float))
    base_formula = BASE + (NAMELEN if args.control_namelen else "")
    baseline = fit(base_formula, panel)
    print(f"panel: {len(panel):,} rows, {int(panel['forgot'].sum()):,} events")
    print(f"baseline: exposure"
          f"{' + NAME LENGTH' if args.control_namelen else ''}\n")

    def chi2_for(idx) -> float:
        p = panel.copy()
        p["ss"], p["os"] = strain_columns(idx, p["id"], meta)
        return LR.between(fit(base_formula + " + ss + os", p), baseline).chi2

    observed = chi2_for(real)
    print(f"real token index      chi2 = {observed:.1f}\n")

    null = []
    for seed in range(args.n):
        null.append(chi2_for(permuted_index(universe, real, seed)))
        print(f"  permutation {seed:>3}    chi2 = {null[-1]:7.1f}", flush=True)

    null = np.asarray(null)
    # +1 in numerator and denominator: the standard finite-sample permutation
    # p-value, which cannot report an impossible p = 0.
    p_emp = (1 + int((null >= observed).sum())) / (1 + len(null))
    print(f"\nnull distribution over {len(null)} permutations")
    print(f"  min {null.min():.1f} | median {np.median(null):.1f} | "
          f"max {null.max():.1f} | mean {null.mean():.1f} (sd {null.std():.1f})")
    print(f"  permutations scoring >= the real index: "
          f"{int((null >= observed).sum())} of {len(null)}")
    print(f"  empirical p = {p_emp:.3f}")
    print("\n  " + ("real index is INDISTINGUISHABLE from scrambled -- the feature "
                    "does not depend on token identity"
                    if p_emp > 0.05 else
                    "real index beats the null -- token identity carries signal"))

    import json
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "observed": float(observed),
        "null": [float(x) for x in null],
        "p_empirical": float(p_emp),
        "n_rows": int(len(panel)),
        "n_events": int(panel["forgot"].sum()),
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
