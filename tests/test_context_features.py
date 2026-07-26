"""Tests for context.property_mix_features on a hand-built synthetic schedule.

No GPU and no dataset download: we build a tiny 3-property fixture and drive the
window logic directly, checking the invariants the plan calls out --

  * shares sum to 1; self-exclusion works
  * window boundaries correct (inclusive/exclusive edges)
  * clipping at a checkpoint never counts steps > S
  * prefix-sum counts match a brute-force recount
  * many-to-many chunks counted once per fact, not once per chunk

Run:  python NLPFinalProject/tests/test_context_features.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import context as C


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #
# 5 facts over 3 properties (A, B, C).  chunk 10 is shared by f1(A) and f5(B):
# a many-to-many chunk that must contribute 2 exposures (one per fact), not 1.
#
#   fid  prop  shared_chunks
#   f1   A     [10]
#   f2   B     [11]
#   f3   C     [12]
#   f4   A     [13]
#   f5   B     [10]
#
# chunk -> step:  10->0, 11->2, 12->4, 13->6   (T = 10 steps, buckets vary)
GROUPS = {"A": "g1", "B": "g1", "C": "g2"}
CHUNK_TO_STEPS = {10: [0], 11: [2], 12: [4], 13: [6]}
STEP_BUCKET = np.array([64, 64, 128, 128, 256, 256, 512, 512, 64, 64], dtype=np.int32)


def build():
    df = pd.DataFrame(
        {"prop": ["A", "B", "C", "A", "B"],
         "shared_chunks": [[10], [11], [12], [13], [10]]},
        index=["f1", "f2", "f3", "f4", "f5"])
    return C.build_property_schedule(df, CHUNK_TO_STEPS, STEP_BUCKET, GROUPS)


def approx(a, b):
    return (a == b) or (isinstance(a, float) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))


# Raw ground-truth exposures (fid, prop, step, bucket) for brute-force checks.
EXPOSURES = [
    ("f1", "A", 0, 64), ("f5", "B", 0, 64),      # chunk 10 (many-to-many)
    ("f2", "B", 2, 128),
    ("f3", "C", 4, 256),
    ("f4", "A", 6, 512),
]


def brute_counts(fid, lo, hi):
    """Self-excluded background prop->count over inclusive step range [lo, hi]."""
    out = {}
    for efid, prop, step, _ in EXPOSURES:
        if lo <= step <= hi and efid != fid:
            out[prop] = out.get(prop, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_many_to_many_and_self_exclusion():
    sched = build()
    # f1 (A, step 0), local W=1 -> window [0,1].  chunk 10 backs f1(A, own) and
    # f5(B): own A excluded, f5's B kept -> background is exactly {B:1}.
    f = C.property_mix_features(sched, "f1", "local", W=1, clip_at=9)
    assert approx(f["count_A"], 0.0), f["count_A"]        # own excluded
    assert approx(f["count_B"], 1.0), f["count_B"]        # co-listed fact kept
    assert approx(f["total"], 1.0)
    assert approx(f["share_B"], 1.0)
    assert approx(f["own_appearances"], 1.0) and f["rehearsed"] == 1.0
    print("ok  many-to-many + self-exclusion")


def test_shares_sum_to_one_and_groups():
    sched = build()
    # f3 (C, step 4), local W=2 -> window [2,6].  exposures: f2(B)@2, f3(C,own)@4,
    # f4(A)@6.  Self-exclude own C -> background {A:1, B:1}, total 2.
    f = C.property_mix_features(sched, "f3", "local", W=2, clip_at=9)
    assert approx(f["count_A"], 1.0) and approx(f["count_B"], 1.0)
    assert approx(f["count_C"], 0.0)
    assert approx(f["total"], 2.0)
    share_sum = f["share_A"] + f["share_B"] + f["share_C"]
    assert approx(share_sum, 1.0), share_sum
    # groups: A,B -> g1 ; C -> g2.  g1 share = 1.0, g2 = 0.0
    assert approx(f["gshare_g1"], 1.0) and approx(f["gshare_g2"], 0.0)
    assert approx(f["gcount_g1"], 2.0)
    # entropy of {0.5, 0.5} = ln 2
    assert approx(f["entropy_prop"], math.log(2)), f["entropy_prop"]
    print("ok  shares sum to 1 + group aggregation + entropy")


def test_since_last_edges_and_bucket():
    sched = build()
    # f3 (C, step 4), since_last -> (4, 9].  Only f4(A)@6 -> {A:1}.
    f = C.property_mix_features(sched, "f3", "since_last", clip_at=9)
    assert approx(f["count_A"], 1.0) and approx(f["total"], 1.0)
    assert approx(f["own_appearances"], 0.0)              # self-excluding by design
    # window_mean_bucket = bucket of step 6 = 512 ; own_mean_bucket = step 4 = 256
    assert approx(f["window_mean_bucket"], 512.0), f["window_mean_bucket"]
    assert approx(f["own_mean_bucket"], 256.0), f["own_mean_bucket"]
    print("ok  since_last edges (exclusive low, inclusive high) + length buckets")


def test_clipping_excludes_future():
    sched = build()
    # Clip f3 at step 3: its only appearance is at step 4 > 3 -> not yet seen ->
    # all-NaN bundle, and crucially nothing past step 3 is ever counted.
    f = C.property_mix_features(sched, "f3", "local", W=5, clip_at=3)
    assert f["total"] == 0.0
    assert math.isnan(f["share_A"])
    # A fact seen by the clip: f1@0 with local W=100 clipped at 1 -> window [0,1],
    # must not reach f2@2.
    g = C.property_mix_features(sched, "f1", "local", W=100, clip_at=1)
    assert approx(g["count_B"], 1.0) and approx(g["total"], 1.0)  # only f5@0
    print("ok  clipping never counts steps > checkpoint")


def test_prefix_matches_bruteforce():
    sched = build()
    # Compare the O(1) prefix-sum counts to a direct recount over several windows.
    cases = [("f1", "local", 1, 9), ("f3", "local", 2, 9), ("f4", "local", 3, 9),
             ("f2", "since_last", None, 9)]
    for fid, window, W, clip in cases:
        f = C.property_mix_features(sched, fid, window, W=W, clip_at=clip)
        # reconstruct the window's inclusive step span for the brute-force recount
        ap = sched.app_steps[fid]
        seen = ap[ap <= clip]
        if window == "local":
            lo, hi = max(0, int(seen.min()) - W), min(clip, int(seen.max()) + W)
        else:  # since_last single interval (last, clip]
            lo, hi = int(seen[-1]) + 1, clip
        want = brute_counts(fid, lo, hi)
        for p in ("A", "B", "C"):
            got = f[f"count_{p}"]
            assert approx(got if not math.isnan(got) else 0.0, float(want.get(p, 0))), \
                (fid, window, p, got, want)
    print("ok  prefix-sum counts == brute-force recount")


def test_checkpoint_composition_descriptor():
    sched = build()
    comp = C.checkpoint_property_composition(sched, checkpoint_steps=[1, 5, 9])
    # 3 properties x 3 checkpoints
    assert comp.shape[0] == 9
    # by step 9 all 5 exposures counted: A:2 (f1,f4), B:2 (f2,f5), C:1 (f3)
    last = comp[comp["checkpoint"] == 9].set_index("property")["cum_count"]
    assert last["A"] == 2 and last["B"] == 2 and last["C"] == 1
    # shares sum to 1 at each checkpoint that has any exposures
    for S in (5, 9):
        s = comp[comp["checkpoint"] == S]["cum_share"].sum()
        assert approx(s, 1.0), (S, s)
    print("ok  checkpoint_property_composition descriptor")


ALL = [test_many_to_many_and_self_exclusion, test_shares_sum_to_one_and_groups,
       test_since_last_edges_and_bucket, test_clipping_excludes_future,
       test_prefix_matches_bruteforce, test_checkpoint_composition_descriptor]

if __name__ == "__main__":
    for t in ALL:
        t()
    print("\nAll context-feature tests passed.")
