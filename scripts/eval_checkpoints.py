"""Step A.1 - Acquisition timeline.

Sweep every checkpoint, evaluate recall (substring match) + gold-answer
probability for every fact, and write one parquet per checkpoint to
``artifacts/recall/``.  Run once per checkpoint (ideally as a Slurm array job,
one GPU each) or loop locally.

Usage:
    python scripts/eval_checkpoints.py                 # all checkpoints
    python scripts/eval_checkpoints.py --steps 0 10000 # a subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import data as D
from src import recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="*", default=config.CHECKPOINT_STEPS)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N facts (smoke test)")
    args = ap.parse_args()

    df = D.load_popqa_df()
    if args.limit:
        df = df.head(args.limit)
    tokenizer = D.load_tokenizer()

    for step in args.steps:
        out_path = config.RECALL_DIR / f"recall_step{step}.parquet"
        if out_path.exists():
            print(f"[skip] {out_path.name} exists")
            continue
        print(f"[load] checkpoint step{step}")
        model = D.load_checkpoint(step)
        res = recall.eval_checkpoint(model, tokenizer, df)
        res["step"] = step
        res.to_parquet(out_path)
        print(f"[write] {out_path}  ({res['correct'].mean():.3f} correct)")
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
