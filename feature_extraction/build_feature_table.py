"""Step A.2 / B - Assemble the feature tables from the recall sweep + corpus.

Produces two artifacts under ``artifacts/features/``:

  * ``panel.parquet``    -- long: one row per (fact, checkpoint) with the
                            time-varying features (timeline + competition) joined
                            to that checkpoint's recall / gold_prob.  Good for
                            trajectory EDA.
  * ``facts.parquet``    -- one row per fact: static content features + learning-
                            dynamics summary + the retention target.  This is
                            what Step C's prediction model trains on.

Requires:
  * artifacts/recall/recall_step*.parquet  (from eval_checkpoints.py)
  * batch_indices  ->  chunk_to_steps cache (built here if missing)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import data as D
from src import schedule
from src.features import timeline, competition, text, learning


# Scalar internal-state columns produced by eval_checkpoints.py (internal.extract_fact).
# We only carry the ones actually present in the recall parquets, so this works
# whether the sweep ran with or without --hidden, on old parquets, or on --limit
# runs.  The per-layer *_by_layer list columns stay in the raw recall parquet only
# (they're for depth-trajectory EDA, not the flat modeling table).
INTERNAL_SCALAR_COLS = [
    "gold_ntokens", "is_top1", "gold_rank", "logit_gap_top", "answer_margin",
    "gold_prob_first", "top1_prob", "gold_surprisal", "pred_entropy",
    "resolution_depth", "resolution_depth_frac", "resid_norm_last",
    "mlp_active_frac_mean", "mlp_active_frac_mean_subj",
    "mlp_participation_mean", "mlp_participation_mean_subj", "subject_found",
]


def make_mention_provider():
    """Build the OLMo-core-backed mention provider, or ``None`` if the tokenized
    corpus / ``olmo_core`` aren't available (the common PopQA-only case)."""
    import glob
    pattern = getattr(config, "TOKENIZED_GLOB", None)
    if not pattern or not glob.glob(pattern):
        return None
    return text.build_olmo_core_provider(
        pattern, work_dir=str(config.DATASET_DIR / "dataset-cache"))


def load_recall_panel(pattern: str = "recall_step*.parquet") -> pd.DataFrame:
    """All recall sweeps under ``RECALL_DIR``, including subdirectories.

    Deliberately a glob rather than ``recall_step{step}.parquet`` over
    ``config.CHECKPOINT_STEPS``: ``eval_checkpoints.py`` tags filenames with the
    run's settings (``_5shot``, ``_onlysameprop``, ``_recallonly``, ...) and
    writes trajectory sweeps into subdirectories, so the exact-name lookup
    silently found nothing for every real run.  The step comes from the file's
    own ``step`` column, so the grid does not have to match ``CHECKPOINT_STEPS``.

    ``.partial.`` files are skipped -- they are mid-run checkpoints holding a
    prefix of the fact set, and concatenating them duplicates rows.
    """
    import glob as _glob

    paths = sorted(p for p in _glob.glob(
        str(config.RECALL_DIR / "**" / pattern), recursive=True)
        if ".partial." not in os.path.basename(p))
    frames, skipped = [], []
    for p in paths:
        try:
            frames.append(pd.read_parquet(p))
        except Exception as e:                       # noqa: BLE001
            skipped.append(f"{os.path.basename(p)}: {type(e).__name__}")
    if skipped:
        print(f"[recall] skipped {len(skipped)} unreadable parquet(s): "
              + "; ".join(skipped)
              + "\n[recall] (a 'Repetition level histogram size mismatch' here "
                "means pyarrow is older than the writer -- upgrade to >=21)")
    if not frames:
        raise FileNotFoundError(
            f"No readable {pattern} under {config.RECALL_DIR}. "
            "Run eval_checkpoints.py first."
        )
    return pd.concat(frames, ignore_index=True)


def main():
    df = D.load_popqa_df().set_index("id", drop=False)
    tokenizer = D.load_tokenizer()
    recall_panel = load_recall_panel()
    chunk_to_steps = schedule.build_chunk_to_steps()

    # Internal-state columns present in this sweep's parquets (see note above).
    internal_cols = [c for c in INTERNAL_SCALAR_COLS if c in recall_panel.columns]
    # (id, step) -> internal state, for the "state at first-learned step" lookup.
    # We deliberately use the FIRST-LEARNED checkpoint, NOT the final one: internal
    # state at the final step would leak `retained` (which is defined by final
    # correctness), exactly like the leaky_* columns.  State at first-learned is a
    # legitimate predictor of whether the fact will stick (cf. confidence_at_first_known).
    internal_by_step = (recall_panel.set_index(["id", "step"])[internal_cols]
                        if internal_cols else None)
    if internal_cols:
        print(f"[internal] carrying {len(internal_cols)} internal-state columns: "
              f"{', '.join(internal_cols)}")
    else:
        print("[internal] no internal-state columns found "
              "(recall parquets predate eval_checkpoints.py's feature pass)")

    # Pre-compute each fact's appearance steps (from shared_chunks).
    appearances = {
        fid: schedule.steps_for_chunks(row["shared_chunks"], chunk_to_steps)
        for fid, row in df.iterrows()
    }

    # --- per-fact learning summary (needs the full trajectory) ---------------
    learn_rows, targets = {}, {}
    for fid, row in df.iterrows():
        traj = (recall_panel[recall_panel["id"] == fid]
                [["step", "correct", "gold_prob"]].to_dict("records"))
        learn_rows[fid] = learning.learning_features(traj, appearances[fid])
        targets[fid] = learning.retention_target(traj)

    # ===================== long panel (fact x checkpoint) ====================
    panel_records = []
    for _, rec in tqdm(recall_panel.iterrows(), total=len(recall_panel),
                       desc="panel"):
        fid, step = rec["id"], rec["step"]
        row = df.loc[fid]
        first_learned = learn_rows[fid]["first_learned_step"]
        r = {"id": fid, "step": step,
             "correct": rec["correct"], "gold_prob": rec["gold_prob"]}
        r.update({c: rec[c] for c in internal_cols})     # internal state at this step
        r.update(timeline.timeline_features(appearances[fid], step))
        r.update(competition.competition_features(row, chunk_to_steps, step,
                                                  first_learned_step=first_learned))
        panel_records.append(r)
    panel = pd.DataFrame(panel_records)
    panel.to_parquet(config.FEATURE_DIR / "panel.parquet")
    print(f"[write] panel.parquet  {panel.shape}")

    # Mention features need LMEnt's tokenized corpus + olmo_core; None if absent
    # (PopQA-only setup) -> we emit NaN placeholders so the columns still exist.
    mention_provider = make_mention_provider()
    if mention_provider is None:
        print("[mentions] tokenized corpus / olmo_core not available "
              "-> mention features emitted as NaN")

    # ===================== per-fact modeling table ===========================
    fact_records = []
    for fid, row in tqdm(df.iterrows(), total=len(df), desc="facts"):
        r = {"id": fid, "subj": row["subj"], "obj": row["obj"], "prop": row["prop"],
             "retained": targets[fid]}
        # static content features (evaluated at the FINAL checkpoint)
        r.update(competition.competition_features(
            row, chunk_to_steps, config.FINAL_STEP,
            first_learned_step=learn_rows[fid]["first_learned_step"]))
        r.update(text.cheap_text_features(row, tokenizer))
        r.update(text.mention_features(row, mention_provider)
                 if mention_provider is not None else text.empty_mention_features())
        r.update(timeline.timeline_features(appearances[fid], config.FINAL_STEP))
        r.update(learn_rows[fid])
        # internal state at the FIRST-LEARNED checkpoint ("how was it stored when it
        # was first acquired") -- a non-leaky snapshot; NaN for never-learned facts.
        sl = learn_rows[fid]["first_learned_step"]
        if internal_by_step is not None and pd.notna(sl) and (fid, int(sl)) in internal_by_step.index:
            r.update({f"atlearn_{c}": v
                      for c, v in internal_by_step.loc[(fid, int(sl))].to_dict().items()})
        fact_records.append(r)
    facts = pd.DataFrame(fact_records)
    facts.to_parquet(config.FEATURE_DIR / "facts.parquet")
    print(f"[write] facts.parquet  {facts.shape}")
    print(f"        retained={facts['retained'].sum()}  "
          f"forgotten={(facts['retained'] == 0).sum()}  "
          f"never-learned={facts['retained'].isna().sum()}")


if __name__ == "__main__":
    main()
