"""Build the (fact, window) panel of token-interference features.

The outcome file is a **parameter**, never hard-coded: pass any trajectory CSV
whose columns are ``id`` plus one column per checkpoint step.  The features
themselves come only from ``popqa-kas`` and ``batch_indices_epoch_*.npy``, so
swapping the few-shot mode (``only-same-prop`` -> ``drop-same-prop``) means
re-running the *analysis*, not rebuilding the features.  The step grid is read
from the CSV header rather than a literal list, so a different checkpoint set
works without edits.

Panel rows are (fact, window) pairs where the fact was **correct at the window's
start** -- the only facts genuinely at risk of being forgotten.  Window ``w``
spans ``(steps[w], steps[w+1]]``.

Usage:
    python feature_extraction/build_strain_panel.py \
        --trajectory 5_shot_trajectory_with_fact_properites.csv \
        --out data/artifacts/features/strain_panel.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.features import strain as S

POPQA_GLOB = os.path.expanduser(
    "~/.cache/huggingface/datasets/dhgottesman___popqa-kas/**/*.arrow")

POPQA_COLS = ["id", "prop", "subj", "obj", "s_uri", "o_uri", "s_pop", "o_pop",
              "subject_chunks", "answer_chunks", "shared_chunks",
              "num_shared_chunks"]


def load_popqa(cols=POPQA_COLS) -> pd.DataFrame:
    files = sorted(glob.glob(POPQA_GLOB, recursive=True))
    if not files:
        raise FileNotFoundError(f"no popqa-kas arrow shards under {POPQA_GLOB}")
    tabs = [pa.ipc.open_stream(pa.memory_map(f)).read_all().select(cols)
            for f in files]
    return pa.concat_tables(tabs).to_pandas()


def step_columns(traj: pd.DataFrame) -> list[str]:
    """Checkpoint columns, in step order, inferred from the header."""
    steps = [c for c in traj.columns if str(c).strip().lstrip("-").isdigit()]
    return sorted(steps, key=lambda c: int(c))


def trajectory_from_recall_dir(recall_dir: str, drop_excluded: bool = True) -> pd.DataFrame:
    """Pivot a directory of ``recall_step*.parquet`` into the wide trajectory form.

    Same shape a trajectory CSV has: an ``id`` column plus one boolean column per
    checkpoint step.  Lets a raw eval sweep be analysed without first hand-making
    a CSV, and takes the step from each file's own ``step`` column rather than the
    filename (sweeps arrive with names like ``recall_step50000_15shot_top10pct
    (2).parquet``).

    Only facts present at **every** checkpoint are kept -- a fact missing from one
    checkpoint would otherwise silently become a gap in its trajectory, and the
    at-risk/forgot construction reads consecutive checkpoints.

    ``drop_excluded`` removes facts flagged by ``lment_popqa.is_excluded`` (the
    answer, or an alias of it, occurs inside the question).  Those are answerable
    by copying from the prompt, so their "correct" says nothing about recall and
    their flips say nothing about forgetting.
    """
    files = sorted(glob.glob(os.path.join(recall_dir, "**", "recall_step*.parquet"),
                             recursive=True))
    files = [f for f in files if ".partial." not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"no recall_step*.parquet under {recall_dir}")

    frames = []
    for f in files:
        cols = ["id", "step", "correct", "excluded"]
        d = pd.read_parquet(f, columns=cols)
        if drop_excluded:
            d = d[~d["excluded"].astype(bool)]
        frames.append(d[["id", "step", "correct"]])
    long = pd.concat(frames, ignore_index=True)

    wide = (long.pivot_table(index="id", columns="step", values="correct",
                            aggfunc="max")
            .rename(columns=lambda s: str(int(s))))
    complete = wide.dropna(axis=0, how="any")
    if len(complete) < len(wide):
        print(f"[trajectory] dropped {len(wide) - len(complete)} facts missing "
              f"from at least one of the {wide.shape[1]} checkpoints")
    return complete.astype(bool).reset_index()


def epoch_paths() -> list[Path]:
    base = config.DATASET_DIR / "dataset-cache"
    paths = [base / f"batch_indices_epoch_{e}.npy" for e in range(1, 7)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing per-epoch batch indices: {missing}. All six are needed -- "
            "epoch 1 alone covers ~5% of the forgetting events."
        )
    return paths


def build(traj_path: str, out_path: str, df_cutoff: int | None,
          idf: bool, use_bpe: bool, recall_dir: str | None = None,
          drop_excluded: bool = True) -> pd.DataFrame:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    if recall_dir:
        traj = trajectory_from_recall_dir(recall_dir, drop_excluded=drop_excluded)
        log(f"trajectory source: {recall_dir} (drop_excluded={drop_excluded})")
    else:
        traj = pd.read_csv(traj_path)
        log(f"trajectory source: {traj_path}")
    steps = step_columns(traj)
    if len(steps) < 2:
        raise ValueError(f"need >=2 step columns in {traj_path}, found {steps}")
    for c in steps:
        traj[c] = traj[c].astype(str).str.lower().isin(["true", "1"])
    log(f"trajectory: {len(traj)} facts x {len(steps)} checkpoints {steps[0]}..{steps[-1]}")

    df = load_popqa()
    log(f"popqa-kas: {len(df)} facts")

    universe = S.EntityUniverse.from_popqa(df)
    log(f"universe: {len(universe)} entities")

    tokenizer = None
    if use_bpe:
        from src.data import load_tokenizer
        tokenizer = load_tokenizer()
        log(f"tokenizer: {tokenizer.__class__.__name__}")
    qid_tokens = universe.tokens(tokenizer)
    index = S.TokenIndex(universe, qid_tokens, df_cutoff=df_cutoff,
                         idf_weighted=idf)
    log(f"token index: {len(index.token_to_qids)} tokens kept "
        f"(cutoff={df_cutoff}, idf={idf})")

    # -- static features ---------------------------------------------------- #
    # NAME LENGTH IS NOT OPTIONAL.  An entity with a k-token name gets k chances
    # to collide, so strain rises steeply with name length (mean strain by token
    # count: 4.8 / 28 / 72 / 109 / 127 for 1..5 tokens; spearman 0.61).  A null
    # control with randomly reassigned tokens reproduced the entire "strain"
    # effect, and controlling name length reverses its sign -- so these columns
    # must be carried and conditioned on, or the strain features are just a
    # roundabout measurement of how long the subject and object strings are.
    static = {}
    for r in df.itertuples():
        s_own = index.ownership(r.s_uri, r.num_shared_chunks)
        o_own = index.ownership(r.o_uri, r.num_shared_chunks)
        static[r.id] = {
            "subj_strain": index.strain(r.s_uri, exclude={r.o_uri}),
            "obj_strain": index.strain(r.o_uri, exclude={r.s_uri}),
            "s_own_rare": s_own["own_rare"], "o_own_rare": o_own["own_rare"],
            "s_rivals_rare": s_own["rivals_rare"],
            "o_rivals_rare": o_own["rivals_rare"],
            "s_ntok": len(index.qid_to_tokens.get(r.s_uri, ())),
            "o_ntok": len(index.qid_to_tokens.get(r.o_uri, ())),
            "s_charlen": len(str(r.subj)),
            "o_charlen": len(str(r.obj)),
        }
    log("static strain + ownership done")

    alt_static = S.alt_answer_features(df, universe)
    log("static alt-answer done")

    # -- windows ------------------------------------------------------------ #
    boundaries = [int(c) for c in steps]
    wm = S.WindowMap.from_batch_indices(epoch_paths(), boundaries)
    log(f"window map built: {wm.n_windows} windows, "
        f"{wm.mask.size} chunk slots")

    rows = []
    meta = df.set_index("id")
    for w in range(wm.n_windows):
        lo, hi = steps[w], steps[w + 1]
        at_risk = traj[traj[lo]]
        active = wm.active_entities(universe, w)
        cfilt = wm.filter_for(w)
        alt_w = S.alt_answer_features(df, universe, chunk_filter=cfilt)
        log(f"  window {w} ({lo}->{hi}): {len(at_risk)} at risk, "
            f"{len(active)} entities active")

        for fid, forgot in zip(at_risk["id"], ~at_risk[hi]):
            if fid not in meta.index:
                continue
            m = meta.loc[fid]
            st = static[fid]
            aw, a0 = alt_w[fid], alt_static[fid]
            rows.append({
                "id": fid, "prop": m["prop"], "window": w,
                "step_lo": int(lo), "step_hi": int(hi),
                "forgot": int(forgot),
                # static
                **st,
                "shared_static": a0["shared"], "distract_static": a0["distract"],
                "evidence_static": a0["evidence"], "ratio_static": a0["ratio"],
                # windowed -- the forms with a time dimension
                "subj_strain_w": S.windowed_strain(index, m["s_uri"], active,
                                                   exclude={m["o_uri"]}),
                "obj_strain_w": S.windowed_strain(index, m["o_uri"], active,
                                                  exclude={m["s_uri"]}),
                "shared_w": aw["shared"], "distract_w": aw["distract"],
                "evidence_w": aw["evidence"], "ratio_w": aw["ratio"],
                # controls
                "s_pop": m["s_pop"], "o_pop": m["o_pop"],
                "num_shared_chunks": m["num_shared_chunks"],
            })

    panel = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    log(f"wrote {out_path}: {len(panel)} rows, "
        f"{int(panel['forgot'].sum())} forgetting events, "
        f"{panel['id'].nunique()} facts")
    return panel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trajectory",
                     help="CSV with an `id` column plus one column per checkpoint step")
    src.add_argument("--recall-dir",
                     help="directory of recall_step*.parquet from an eval sweep; "
                          "pivoted into the same wide form")
    ap.add_argument("--keep-excluded", action="store_true",
                    help="keep facts whose answer occurs in the question "
                         "(--recall-dir only; they are dropped by default)")
    ap.add_argument("--out", default=str(config.FEATURE_DIR / "strain_panel.parquet"))
    ap.add_argument("--df-cutoff", type=int, default=S.DF_CUTOFF,
                    help="drop tokens carried by more than this many entities "
                         "(-1 to keep all)")
    ap.add_argument("--idf", action="store_true",
                    help="IDF-weight shared tokens instead of counting them")
    ap.add_argument("--bpe", action="store_true",
                    help="tokenize with the model tokenizer instead of words")
    a = ap.parse_args()
    build(a.trajectory, a.out, None if a.df_cutoff < 0 else a.df_cutoff,
          a.idf, a.bpe, recall_dir=a.recall_dir,
          drop_excluded=not a.keep_excluded)


if __name__ == "__main__":
    main()
