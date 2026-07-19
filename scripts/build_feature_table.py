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


def load_recall_panel() -> pd.DataFrame:
    frames = []
    for step in config.CHECKPOINT_STEPS:
        p = config.RECALL_DIR / f"recall_step{step}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError("No recall_step*.parquet found. Run eval_checkpoints.py first.")
    return pd.concat(frames, ignore_index=True)


def main():
    df = D.load_popqa_df().set_index("id", drop=False)
    tokenizer = D.load_tokenizer()
    recall_panel = load_recall_panel()
    chunk_to_steps = schedule.build_chunk_to_steps()

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
        r.update(timeline.timeline_features(appearances[fid], step))
        r.update(competition.competition_features(row, chunk_to_steps, step,
                                                  first_learned_step=first_learned))
        panel_records.append(r)
    panel = pd.DataFrame(panel_records)
    panel.to_parquet(config.FEATURE_DIR / "panel.parquet")
    print(f"[write] panel.parquet  {panel.shape}")

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
        r.update(timeline.timeline_features(appearances[fid], config.FINAL_STEP))
        r.update(learn_rows[fid])
        fact_records.append(r)
    facts = pd.DataFrame(fact_records)
    facts.to_parquet(config.FEATURE_DIR / "facts.parquet")
    print(f"[write] facts.parquet  {facts.shape}")
    print(f"        retained={facts['retained'].sum()}  "
          f"forgotten={(facts['retained'] == 0).sum()}  "
          f"never-learned={facts['retained'].isna().sum()}")


if __name__ == "__main__":
    main()
