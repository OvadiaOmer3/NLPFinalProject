"""Group 5 - "What else was being trained" (property-background / propmix).

Generalizes ``competition.py``'s fact-relative window from "other chunks about the
*same entity*" to "**other facts of other relation types** trained nearby".  For a
fact we look at a window of training time around it and summarize the property mix
of the *other* facts trained in that window -- so we can ask e.g. "is an
``occupation`` fact retained less when many ``genre`` facts were trained around it?"

Scoping / counting rules (see the plan's decisions 3-7):

* **Counting unit = (fact, step) exposures over ``shared_chunks`` only.**
  ``shared_chunks`` are the chunks where the relation is actually instantiated
  (subject *and* answer co-occur).  chunk->property is many-to-many (one passage may
  state occupation *and* place-of-birth), so we count fact-exposures, not chunks: a
  chunk backing 3 facts contributes 3 exposures.  This keeps the shares summing to 1.
* **Self-exclusion.** A fact's *own* exposures never count toward its own background
  (otherwise the own-property share is mechanically tied to ``n_appearances``).
* **Coverage caveat.** ~17% of PopQA-KAS facts (2,450/14,267) have no shared chunks
  and get an all-NaN bundle; and the annotated slice is far smaller than the corpus.
  ``propmix`` means *"among the trackable factual content near this fact, what was
  the property mix"* -- NOT "what fraction of training tokens were about genre".

Everything derives from ``batch_indices`` + PopQA ``shared_chunks`` (+ the per-step
length bucket): no model, no outcome, no cross-fact normalization.  Shares are the
only normalization and they are strictly within-row, so the emitted table is
leakage-safe under any later train/test split.

Windows (all fact-anchored, so two facts in the same checkpoint bucket still differ):

* ``local``      -- union over the fact's appearance steps ``a`` of ``[a-W, a+W]``.
                    "What was trained *alongside* it."  Primary for identification
                    (Step 0: highest within-property residual variance).
* ``since_last`` -- ``(last_seen_step, checkpoint]``.  "What was trained *since* the
                    fact was last rehearsed."  Self-excluding by construction.
* ``postlearn``  -- ``(first_learned_step, checkpoint]`` (``anchor=first_learned``).
                    Mirrors ``competition.py``'s post-learn interference window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------- #
# Relation -> type family (decision 6): the hypothesis is about entity *types*.
# --------------------------------------------------------------------------- #
PROPERTY_GROUPS = {
    "author": "creative_work",
    "composer": "creative_work",
    "director": "creative_work",
    "genre": "creative_work",
    "producer": "creative_work",
    "screenwriter": "creative_work",
    "father": "person",
    "mother": "person",
    "occupation": "person",
    "place of birth": "person",
    "religion": "person",
    "sport": "person",
    "capital": "place",
    "capital of": "place",
    "country": "place",
    "color": "misc",
}


def _san(name: str) -> str:
    """Column-safe property name (``"place of birth" -> "place_of_birth"``)."""
    return name.replace(" ", "_")


# --------------------------------------------------------------------------- #
# Precomputed schedule
# --------------------------------------------------------------------------- #
@dataclass
class PropertySchedule:
    prop_names: list          # index order for the K per-property columns
    group_names: list         # index order for the G per-group columns
    prop_to_group: np.ndarray # (K,) group index for each property
    nsteps: int               # number of optimizer steps T
    prop_prefix: np.ndarray   # (K, T+1) cumulative per-property exposure counts
    count_prefix: np.ndarray  # (T+1,)   cumulative total exposure counts
    bucket_prefix: np.ndarray # (T+1,)   cumulative sum of exposure length-buckets
    app_steps: dict           # fid -> sorted np.ndarray of appearance steps
    app_buckets: dict         # fid -> np.ndarray of length-buckets, aligned to app_steps
    fact_prop: dict           # fid -> property index

    @property
    def K(self) -> int:
        return len(self.prop_names)

    @property
    def G(self) -> int:
        return len(self.group_names)


def build_property_schedule(df, chunk_to_steps, step_bucket,
                            property_groups=PROPERTY_GROUPS) -> PropertySchedule:
    """Precompute per-property prefix sums so any window count is O(1) per property.

    ``df`` is the PopQA table (index = fact id, columns include ``prop`` and
    ``shared_chunks``).  ``chunk_to_steps`` maps chunk id -> its appearance steps
    (``schedule.build_chunk_to_steps``).  ``step_bucket`` is the per-step length
    bucket (``schedule.build_step_bucket``).
    """
    prop_names = sorted(df["prop"].unique())
    p2i = {p: i for i, p in enumerate(prop_names)}
    group_names = sorted({property_groups.get(p, p) for p in prop_names})
    g2i = {g: i for i, g in enumerate(group_names)}
    prop_to_group = np.array([g2i[property_groups.get(p, p)] for p in prop_names],
                             dtype=np.int64)

    T = int(len(step_bucket))
    step_bucket = np.asarray(step_bucket)

    # (fact, prop, step) exposures over shared_chunks -- one per (fact, chunk, step).
    fids, props, steps = [], [], []
    fact_prop = {}
    for fid, row in df.iterrows():
        pi = p2i[row["prop"]]
        fact_prop[fid] = pi
        for c in row["shared_chunks"]:
            for s in chunk_to_steps.get(int(c), ()):
                fids.append(fid); props.append(pi); steps.append(s)
    steps = np.asarray(steps, dtype=np.int64)
    props = np.asarray(props, dtype=np.int64)
    buckets = step_bucket[steps].astype(np.int64)
    K = len(prop_names)

    # Per-property and total per-step counts -> prefix sums (prefix[t] = sum steps < t).
    prop_counts = np.bincount(props * T + steps, minlength=K * T).reshape(K, T)
    prop_prefix = np.zeros((K, T + 1), dtype=np.int64)
    np.cumsum(prop_counts, axis=1, out=prop_prefix[:, 1:])
    count_prefix = np.zeros(T + 1, dtype=np.int64)
    np.cumsum(np.bincount(steps, minlength=T), out=count_prefix[1:])
    bucket_prefix = np.zeros(T + 1, dtype=np.int64)
    np.cumsum(np.bincount(steps, weights=buckets, minlength=T).astype(np.int64),
              out=bucket_prefix[1:])

    # Per-fact sorted appearance steps + aligned buckets.
    app_steps, app_buckets = {}, {}
    fid_arr = np.asarray(fids, dtype=object)
    order = np.argsort(steps, kind="stable")
    s_sorted, b_sorted, f_sorted = steps[order], buckets[order], fid_arr[order]
    exp = pd.DataFrame({"fid": f_sorted, "step": s_sorted, "bucket": b_sorted})
    for fid, g in exp.groupby("fid", sort=False):
        app_steps[fid] = g["step"].to_numpy()
        app_buckets[fid] = g["bucket"].to_numpy()

    return PropertySchedule(prop_names, group_names, prop_to_group, T,
                            prop_prefix, count_prefix, bucket_prefix,
                            app_steps, app_buckets, fact_prop)


# --------------------------------------------------------------------------- #
# Window features
# --------------------------------------------------------------------------- #
def _merge_intervals(los, his):
    """Merge (possibly overlapping) inclusive integer intervals."""
    order = np.argsort(los)
    los, his = los[order], his[order]
    out = []
    cl, ch = int(los[0]), int(his[0])
    for a, b in zip(los[1:], his[1:]):
        a, b = int(a), int(b)
        if a <= ch + 1:
            ch = max(ch, b)
        else:
            out.append((cl, ch)); cl, ch = a, b
    out.append((cl, ch))
    return out


def _window_intervals(seen, window, W, hi_cap, anchor):
    """Inclusive step intervals for a window, all clipped to ``[0, hi_cap]``.

    ``seen`` = the fact's appearance steps up to the checkpoint (sorted).
    """
    if window == "local":
        los = np.clip(seen - W, 0, hi_cap)
        his = np.clip(seen + W, 0, hi_cap)
        return _merge_intervals(los, his)
    if window == "since_last":
        last = int(seen[-1])
        return [] if last >= hi_cap else [(last + 1, hi_cap)]
    if window == "postlearn":
        lo = int(anchor) + 1
        return [] if lo > hi_cap else [(lo, hi_cap)]
    raise ValueError(f"unknown window {window!r}")


def _empty_features(sched) -> dict:
    f = {}
    for p in sched.prop_names:
        f[f"count_{_san(p)}"] = np.nan
        f[f"share_{_san(p)}"] = np.nan
    for g in sched.group_names:
        f[f"gcount_{g}"] = np.nan
        f[f"gshare_{g}"] = np.nan
    f.update(total=0.0, entropy_prop=np.nan, entropy_grp=np.nan, efflen=0.0,
             own_appearances=0.0, rehearsed=0.0,
             window_mean_bucket=np.nan, own_mean_bucket=np.nan)
    return f


def _entropy(shares) -> float:
    p = shares[shares > 0]
    return float(-(p * np.log(p)).sum()) if p.size else np.nan


def property_mix_features(sched: PropertySchedule, fid, window,
                          W=None, clip_at=None, anchor=None) -> dict:
    """Background property-mix bundle for one fact over one window.

    ``clip_at`` (a checkpoint step) clips the window so it never looks past the
    checkpoint the outcome is measured at; appearances after it are also ignored.
    ``anchor`` is the low bound for ``postlearn`` (the fact's first_learned_step).
    Returns all-NaN (via ``_empty_features``) when the fact has no usable window
    (never seen by the checkpoint, or -- for ``postlearn`` -- never learned).
    """
    if fid not in sched.app_steps:
        return _empty_features(sched)
    hi_cap = sched.nsteps - 1 if clip_at is None else min(int(clip_at), sched.nsteps - 1)

    ap, bk = sched.app_steps[fid], sched.app_buckets[fid]
    mask = ap <= hi_cap
    seen, seen_bk = ap[mask], bk[mask]
    if seen.size == 0:
        return _empty_features(sched)
    if window == "postlearn" and (anchor is None or not np.isfinite(anchor)):
        return _empty_features(sched)

    intervals = _window_intervals(seen, window, W, hi_cap, anchor)
    if not intervals:
        return _empty_features(sched)

    # Background counts (all exposures in the window), then self-exclude own.
    K = sched.K
    counts = np.zeros(K, dtype=np.float64)
    total = 0.0
    bucket_sum = 0.0
    efflen = 0
    for lo, hi in intervals:
        counts += sched.prop_prefix[:, hi + 1] - sched.prop_prefix[:, lo]
        total += sched.count_prefix[hi + 1] - sched.count_prefix[lo]
        bucket_sum += sched.bucket_prefix[hi + 1] - sched.bucket_prefix[lo]
        efflen += hi - lo + 1

    # Own exposures falling inside the window (for self-exclusion + rehearsal).
    in_win = np.zeros(seen.size, dtype=bool)
    for lo, hi in intervals:
        in_win |= (seen >= lo) & (seen <= hi)
    own_n = int(in_win.sum())
    own_bucket_in = float(seen_bk[in_win].sum())
    P = sched.fact_prop[fid]
    counts[P] -= own_n
    total -= own_n
    bucket_sum -= own_bucket_in

    f = {}
    if total > 0:
        shares = counts / total
        gcounts = np.zeros(sched.G)
        np.add.at(gcounts, sched.prop_to_group, counts)
        gshares = gcounts / total
        for i, p in enumerate(sched.prop_names):
            f[f"count_{_san(p)}"] = float(counts[i])
            f[f"share_{_san(p)}"] = float(shares[i])
        for j, g in enumerate(sched.group_names):
            f[f"gcount_{g}"] = float(gcounts[j])
            f[f"gshare_{g}"] = float(gshares[j])
        f["entropy_prop"] = _entropy(shares)
        f["entropy_grp"] = _entropy(gshares)
        f["window_mean_bucket"] = bucket_sum / total
    else:
        # Window exists but holds no background exposures (e.g. no other facts nearby).
        for i, p in enumerate(sched.prop_names):
            f[f"count_{_san(p)}"] = float(max(counts[i], 0.0))
            f[f"share_{_san(p)}"] = np.nan
        for g in sched.group_names:
            f[f"gcount_{g}"] = 0.0
            f[f"gshare_{g}"] = np.nan
        f["entropy_prop"] = np.nan
        f["entropy_grp"] = np.nan
        f["window_mean_bucket"] = np.nan

    f["total"] = float(max(total, 0.0))
    f["efflen"] = float(efflen)
    f["own_appearances"] = float(own_n)
    f["rehearsed"] = float(own_n > 0)
    # own_mean_bucket: mean length bucket of the fact's own exposures up to the
    # checkpoint (window-independent), so it is defined even for empty windows.
    f["own_mean_bucket"] = float(seen_bk.mean())
    return f


# --------------------------------------------------------------------------- #
# Checkpoint-level descriptor (decision 2): a run descriptor, NOT a predictor.
# --------------------------------------------------------------------------- #
def checkpoint_property_composition(sched: PropertySchedule,
                                    checkpoint_steps=config.CHECKPOINT_STEPS) -> pd.DataFrame:
    """Cumulative-from-zero property composition indexed by (property, checkpoint).

    A function of (property, checkpoint) only -- 16 x 12 distinct values, replicated
    across every fact of a property -- so it has zero within-(property, checkpoint)
    variance and cannot be a fact-level predictor.  Written to its own artifact as a
    run descriptor / control (do NOT merge into facts.parquet).
    """
    rows = []
    for S in checkpoint_steps:
        t = min(int(S), sched.nsteps - 1) + 1
        cum = sched.prop_prefix[:, t]
        total = int(cum.sum())
        for i, p in enumerate(sched.prop_names):
            rows.append({"property": p, "checkpoint": int(S),
                         "cum_count": int(cum[i]),
                         "cum_share": (float(cum[i] / total) if total else np.nan)})
    return pd.DataFrame(rows)
