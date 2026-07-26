"""Step 0 -- variance diagnostic (GO / NO-GO gate for the property-background feature).

Question this answers
---------------------
Every fact-level window (``since_last``, ``local``) derives its *background
property composition* (``propmix``) from where the fact sits in training.  If the
corpus had been shuffled iid, between-fact variation in that composition would be
pure multinomial sampling noise ``~ p_k(1-p_k)/n_window`` -- clean (randomly
assigned, no confounding) but possibly far too small to detect against retention
noise at n ~ 14k.  If instead the curriculum groups content in time (LMEnt uses a
VSL ``grow_p2`` 8-cycle curriculum, and batch sizes here already swing 16->512),
the composition carries real structure and the FE controls become essential.

So we compare, per background property k:

    observed dispersion   obs_k = mean_f (share_{f,k} - p_k)^2
    expected (iid) noise  exp_k = mean_f p_k(1-p_k)/n_f
    ratio_k = obs_k / exp_k

and a single pooled overdispersion factor phi (Pearson index of dispersion,
E[phi]=1 under an iid multinomial ordering).  A permutation null (shuffle each
exposure's property label, holding every window and every per-step exposure count
fixed) gives an empirical baseline for ``since_last``; it should land at phi~1 and
validates the analytic reference.

    phi ~ 1   -> batches effectively iid-shuffled: composition is randomly
                 assigned (a clean natural experiment) but effects will be small
                 -> a POWER question.  Weak GO / caution.
    phi >> 1  -> curriculum / content structure exists -> more signal, FE controls
                 become essential.  GO.

Also reports ``phi_within`` (dispersion after centering on each own-property's
mean background mix): the within-(own-property) variation that trap #2 in the plan
needs to survive for the effect to be identified.

Inputs (no GPU, no model): ``batch_indices.npy`` + PopQA-KAS ``shared_chunks``.
Counting unit is (fact, step) exposures over ``shared_chunks`` only, self-excluded
(see the plan's decision #4).  ``since_last`` is self-excluding by construction
(its window is strictly after the fact's last appearance); ``local`` subtracts the
fact's own in-window appearances.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import data as D

# --------------------------------------------------------------------------- #
# Config for the diagnostic
# --------------------------------------------------------------------------- #
LOCAL_WS = (100, 250, 500, 1000, 2000)   # half-widths for the `local` window sweep
N_PERM = 100                # permutation-null replicates (since_last only)
SEED = 0
OUT_DIR = config.FEATURE_DIR / "diagnostics"


# --------------------------------------------------------------------------- #
# Build the exposure table  ((fact, step) events over shared_chunks)
# --------------------------------------------------------------------------- #
def build_exposures(df: pd.DataFrame, batch_indices):
    """Return (exposures_df, nsteps, prop_names, prop_idx_per_fact, appearances).

    exposures_df: one row per (fact, step) event -- a shared_chunk of the fact was
    trained at that step.  Columns: fact_idx (int), prop_idx (int8), step (int32).
    A chunk backing several facts yields one exposure per fact (many-to-many
    counted once per fact, per plan decision #4).
    """
    nsteps = len(batch_indices)
    prop_names = sorted(df["prop"].unique())
    prop_to_idx = {p: i for i, p in enumerate(prop_names)}
    prop_idx_per_fact = df["prop"].map(prop_to_idx).to_numpy(np.int16)

    # (fact_idx, prop_idx, chunk) : one row per (fact, shared_chunk) pair.
    fact_rows, prop_rows, chunk_rows = [], [], []
    for fidx, (_, row) in enumerate(df.iterrows()):
        sc = row["shared_chunks"]
        if len(sc) == 0:
            continue
        pi = int(prop_idx_per_fact[fidx])
        for c in sc:
            fact_rows.append(fidx)
            prop_rows.append(pi)
            chunk_rows.append(int(c))
    shared_pairs = pd.DataFrame(
        {"fact_idx": np.asarray(fact_rows, np.int32),
         "prop_idx": np.asarray(prop_rows, np.int16),
         "chunk": np.asarray(chunk_rows, np.int64)})

    # (chunk, step) for every trained slot whose chunk backs *some* fact.
    # Flatten the variable-width batches into aligned (chunk, step) arrays, then
    # keep only slots whose chunk is a shared_chunk of some fact.
    batch_lens = np.fromiter((len(b) for b in batch_indices), dtype=np.int64,
                             count=nsteps)
    all_chunks = np.concatenate([np.asarray(b).ravel() for b in batch_indices]).astype(np.int64)
    all_steps = np.repeat(np.arange(nsteps, dtype=np.int32), batch_lens)
    shared_set = np.unique(shared_pairs["chunk"].to_numpy())
    keep = np.isin(all_chunks, shared_set, assume_unique=False)
    trained = pd.DataFrame({"chunk": all_chunks[keep], "step": all_steps[keep]})

    # (fact, chunk) x (chunk, step)  ->  (fact, prop, step) exposures.
    exposures = shared_pairs.merge(trained, on="chunk", how="inner")
    exposures = exposures[["fact_idx", "prop_idx", "step"]].reset_index(drop=True)

    # Per-fact sorted appearance steps (with multiplicity).
    appearances = {int(f): np.sort(g["step"].to_numpy())
                   for f, g in exposures.groupby("fact_idx")}
    return exposures, nsteps, prop_names, prop_idx_per_fact, appearances


def prop_step_prefix(exposures, K, nsteps):
    """Per-property cumulative exposure counts: prefix[k, t] = #exposures of k at
    steps < t.  Window count over inclusive step range [lo, hi] is
    prefix[k, hi+1] - prefix[k, lo]."""
    flat = exposures["prop_idx"].to_numpy(np.int64) * nsteps + exposures["step"].to_numpy(np.int64)
    counts = np.bincount(flat, minlength=K * nsteps).reshape(K, nsteps)
    prefix = np.zeros((K, nsteps + 1), dtype=np.int64)
    np.cumsum(counts, axis=1, out=prefix[:, 1:])
    return prefix


# --------------------------------------------------------------------------- #
# Window count matrices  (facts x K), self-excluded
# --------------------------------------------------------------------------- #
def counts_since_last(appearances, prefix, nsteps, fact_props, K):
    """since_last window (last_seen, FINAL].  Self-excluding by construction:
    no own appearance lies strictly after the fact's last one."""
    fids = sorted(appearances)
    lo = np.array([appearances[f][-1] + 1 for f in fids], dtype=np.int64)  # last+1
    # count over [lo, nsteps-1] = prefix[:, nsteps] - prefix[:, lo]
    C = (prefix[:, nsteps][:, None] - prefix[:, lo]).T          # (nfacts, K)
    keep = lo < nsteps                                          # drop empty windows
    C = C[keep]
    used = [f for f, k in zip(fids, keep) if k]
    own = np.array([fact_props[f] for f in used])
    return np.asarray(used), C, own


def _merge_intervals(lo, hi):
    order = np.argsort(lo)
    lo, hi = lo[order], hi[order]
    out = []
    cl, ch = lo[0], hi[0]
    for a, b in zip(lo[1:], hi[1:]):
        if a <= ch + 1:
            ch = max(ch, b)
        else:
            out.append((cl, ch)); cl, ch = a, b
    out.append((cl, ch))
    return out


def counts_local(appearances, prefix, nsteps, fact_props, K, W):
    """local window: union over appearances a of [a-W, a+W], clipped to
    [0, nsteps-1].  Self-exclude the fact's own in-window appearances (all of
    them, since each a sits inside its own [a-W, a+W])."""
    fids = sorted(appearances)
    C = np.zeros((len(fids), K), dtype=np.int64)
    own = np.empty(len(fids), dtype=np.int64)
    for i, f in enumerate(fids):
        ap = appearances[f]
        lo = np.clip(ap - W, 0, nsteps - 1)
        hi = np.clip(ap + W, 0, nsteps - 1)
        cnt = np.zeros(K, dtype=np.int64)
        for a, b in _merge_intervals(lo, hi):
            cnt += prefix[:, b + 1] - prefix[:, a]
        p = fact_props[f]
        cnt[p] -= len(ap)               # self-exclusion
        C[i] = cnt
        own[i] = p
    return np.asarray(fids), C, own


# --------------------------------------------------------------------------- #
# Dispersion statistics
# --------------------------------------------------------------------------- #
def dispersion_stats(C, own, K):
    """All the observed-vs-iid-noise numbers for one window's count matrix."""
    n = C.sum(axis=1)
    ok = n > 0
    C, own, n = C[ok], own[ok], n[ok]
    shares = C / n[:, None]
    p = C.sum(axis=0) / C.sum()                       # global background prevalence

    obs_k = ((shares - p) ** 2).mean(axis=0)
    exp_k = (p * (1 - p))[None, :] / n[:, None]
    exp_k = exp_k.mean(axis=0)
    ratio_k = np.where(exp_k > 0, obs_k / np.where(exp_k > 0, exp_k, 1), np.nan)

    phi = _pearson_phi(C, n, p, K)                    # pooled overdispersion

    # within-(own-property) overdispersion: center shares on each group's mean.
    resid2 = np.zeros_like(shares)
    for g in np.unique(own):
        m = own == g
        resid2[m] = (shares[m] - shares[m].mean(axis=0)) ** 2
    obs_within = resid2.mean(axis=0)
    phi_within = float((obs_within / np.where(exp_k > 0, exp_k, np.nan)).mean())

    # Absolute magnitude of the between-fact variation (the honest power number):
    # how many percentage points does each background share swing, fact to fact?
    obs_sd_pp = np.sqrt(obs_k) * 100.0                # observed between-fact SD
    exp_sd_pp = np.sqrt(exp_k) * 100.0                # iid-noise floor
    return {
        "n_facts": int(len(n)),
        "n_window": {"min": int(n.min()), "median": float(np.median(n)),
                     "mean": float(n.mean()), "max": int(n.max())},
        "prevalence": p.tolist(),
        "obs_var": obs_k.tolist(),
        "exp_var": exp_k.tolist(),
        "obs_sd_pp": obs_sd_pp.tolist(),
        "exp_sd_pp": exp_sd_pp.tolist(),
        "ratio_per_prop": ratio_k.tolist(),
        "phi_pooled": float(phi),
        "phi_within_prop": phi_within,
    }


def _pearson_phi(C, n, p, K):
    exp = n[:, None] * p[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(exp > 0, (C - exp) ** 2 / exp, 0.0)
    X = term.sum(axis=1)                              # per-fact Pearson X^2
    return float(X.mean() / (K - 1))                  # E[X]=K-1 under multinomial


def permutation_phi(exposures, appearances, nsteps, fact_props, K, n_perm, seed):
    """Empirical iid-ordering null for since_last: shuffle each exposure's property
    label, holding every per-step exposure count and every window fixed."""
    rng = np.random.default_rng(seed)
    labels = exposures["prop_idx"].to_numpy(np.int64)
    steps = exposures["step"].to_numpy(np.int64)
    fids = [f for f in sorted(appearances) if appearances[f][-1] + 1 < nsteps]
    lo = np.array([appearances[f][-1] + 1 for f in fids], dtype=np.int64)
    p_global = np.bincount(labels, minlength=K) / len(labels)
    out = []
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        flat = perm * nsteps + steps
        counts = np.bincount(flat, minlength=K * nsteps).reshape(K, nsteps)
        prefix = np.zeros((K, nsteps + 1), dtype=np.int64)
        np.cumsum(counts, axis=1, out=prefix[:, 1:])
        C = (prefix[:, nsteps][:, None] - prefix[:, lo]).T
        n = C.sum(axis=1)
        out.append(_pearson_phi(C[n > 0], n[n > 0], p_global, K))
    return np.array(out)


# --------------------------------------------------------------------------- #
def _fmt_top(names, ratios, k=6):
    order = np.argsort(ratios)[::-1]
    return ", ".join(f"{names[i]}={ratios[i]:.1f}x" for i in order[:k])


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load] PopQA-KAS + batch_indices ...")
    df = D.load_popqa_df().reset_index(drop=True)
    batch_indices = np.load(config.BATCH_INDICES_PATH, allow_pickle=True)

    print("[build] exposure table (fact, step) over shared_chunks ...")
    exposures, nsteps, prop_names, fact_props, appearances = build_exposures(df, batch_indices)
    K = len(prop_names)
    print(f"        {len(exposures):,} exposures | {len(appearances):,} facts with "
          f"shared_chunks | {K} properties | {nsteps:,} steps  "
          f"({time.time()-t0:.0f}s)")

    prefix = prop_step_prefix(exposures, K, nsteps)

    windows = {"since_last": counts_since_last(appearances, prefix, nsteps, fact_props, K)}
    for W in LOCAL_WS:
        windows[f"local_W{W}"] = counts_local(appearances, prefix, nsteps, fact_props, K, W)

    report = {"n_props": K, "prop_names": prop_names, "windows": {}}
    for name, (used, C, own) in windows.items():
        stats = dispersion_stats(C, own, K)
        report["windows"][name] = stats
        print("\n" + "=" * 72)
        print(f"WINDOW: {name}   (facts used: {stats['n_facts']:,})")
        nw = stats["n_window"]
        print(f"  window exposures n_f: min={nw['min']}  median={nw['median']:.0f}  "
              f"mean={nw['mean']:.0f}  max={nw['max']:,}")
        print(f"  phi_pooled (overdispersion, iid=>1) : {stats['phi_pooled']:.2f}x")
        print(f"  phi_within_prop (residual, iid=>1)  : {stats['phi_within_prop']:.2f}x")
        print(f"  top per-property obs/exp variance   : "
              f"{_fmt_top(prop_names, np.array(stats['ratio_per_prop']))}")
        # Absolute swing = real power. iid floor is what we'd see under shuffling.
        sd_obs = np.array(stats["obs_sd_pp"]); sd_exp = np.array(stats["exp_sd_pp"])
        j = int(np.argmax(sd_obs))
        print(f"  ABSOLUTE between-fact share swing   : "
              f"max SD={sd_obs[j]:.2f}pp ({prop_names[j]}), "
              f"median SD={np.median(sd_obs):.3f}pp  "
              f"[iid floor: max={sd_exp.max():.3f}pp]")

    print("\n" + "=" * 72)
    print(f"[permutation null] since_last, {N_PERM} replicates ...")
    perm = permutation_phi(exposures, appearances, nsteps, fact_props, K, N_PERM, SEED)
    obs_phi = report["windows"]["since_last"]["phi_pooled"]
    z = (obs_phi - perm.mean()) / (perm.std() + 1e-12)
    report["permutation"] = {"null_phi_mean": float(perm.mean()),
                             "null_phi_std": float(perm.std()),
                             "observed_phi": float(obs_phi), "z": float(z)}
    print(f"  null phi (iid ordering): {perm.mean():.3f} +/- {perm.std():.3f}   "
          f"[should sit at ~1.0, validating the analytic reference]")
    print(f"  observed phi           : {obs_phi:.2f}x   (z = {z:,.0f} vs null)")

    # ------------------------------------------------------------------- verdict
    sl = report["windows"]["since_last"]
    print("\n" + "#" * 72)
    print("VERDICT")
    print(f"  since_last overdispersion phi = {sl['phi_pooled']:.1f}x iid noise; "
          f"within-property residual = {sl['phi_within_prop']:.1f}x.")
    if sl["phi_pooled"] < 1.5:
        verdict = ("NO-GO / underpowered: window composition is ~iid sampling "
                   "noise. Randomly assigned (clean) but effects will be tiny at "
                   "n~14k -- a power problem, not a confounding one.")
    elif sl["phi_within_prop"] < 1.5:
        verdict = ("WEAK GO: strong between-property structure but little "
                   "within-(own-property) variation -- FE demeaning (trap #2) may "
                   "absorb most of it. Identification will lean on the fine local "
                   "window; check power carefully.")
    else:
        verdict = ("GO: composition carries structure well beyond iid noise, and "
                   "within-property variation survives, so the FE controls are "
                   "meaningful rather than absorbing everything. The feature can "
                   "carry signal.")
    print("  " + verdict)
    report["verdict"] = verdict

    out = OUT_DIR / "variance_diagnostic.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[write] {out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
