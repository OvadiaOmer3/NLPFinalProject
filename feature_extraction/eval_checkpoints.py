"""Step A.1 - Acquisition timeline (+ internal states).

Sweep every checkpoint and, for every fact, record:
  * recall (substring match) + gold-answer probability   -> the acquisition curve
  * internal-state features from a single forward pass    -> how the fact is stored
  * (optional) the full per-layer hidden states in f16    -> probes / drift offline

Writes one enriched parquet per checkpoint to ``artifacts/recall/`` and, unless
``--no-hidden`` is passed, one f16 hidden-state dump per checkpoint to
``artifacts/internal/``.  The recall parquet keeps its original columns (so
``build_feature_table.py`` is unaffected) and gains the internal-feature columns.

Cost model: we touch the model exactly twice per (checkpoint, fact) -- once for
the greedy decode that ``correct`` needs, once for the teacher-forced forward
that yields *everything else* (gold_prob, logit-lens depth, next-token geometry,
per-layer residual / MLP activity, and the hidden-state dump).

Run once per checkpoint (ideally a Slurm array job, one GPU each) or loop locally.

Usage:
    python feature_extraction/eval_checkpoints.py                 # all checkpoints
    python feature_extraction/eval_checkpoints.py --steps 0 10000 # a subset
    python feature_extraction/eval_checkpoints.py --no-hidden     # scalars only, no dump
    python feature_extraction/eval_checkpoints.py --limit 100     # smoke test on 100 facts
    python feature_extraction/eval_checkpoints.py --output-dir D:/lment_out  # explicit output root
                                                                  # (cwd-independent; overrides
                                                                  # $LMENT_DATA_ROOT/artifacts)
    python feature_extraction/eval_checkpoints.py --min-shared-chunks 1  # drop facts that can
                                                                  # never be "seen" (see below)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src import data as D
from src import lment_popqa
from src import recall
from src.features import internal


def eval_step(model, tokenizer, df, step, *, dump_hidden, positions, top_k=None,
             internal_dir=None):
    """Full single-checkpoint pass -> a DataFrame (one row per fact).

    Also streams the f16 hidden-state dump (when ``dump_hidden``) and the top-k
    active-neuron index dump (when ``top_k``) to ``internal_dir`` (defaults to
    ``config.INTERNAL_DIR``, i.e. ``artifacts/internal/``).
    """
    internal_dir = internal_dir or config.INTERNAL_DIR
    ids = df["id"].tolist()
    dump = None
    if dump_hidden:
        dump = internal.HiddenDump(
            step, ids, model.config.num_hidden_layers + 1,
            model.config.hidden_size, positions, internal_dir)
    ndump = None
    if top_k:
        # neurons are captured at answer + subject_last (both slots the hook sees).
        ndump = internal.NeuronDump(
            step, ids, model.config.num_hidden_layers, top_k,
            ("answer", "subject_last"), internal_dir)

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        # (1) greedy decode -> substring-match correctness (paper's `correct`).
        prompt = D.prompt_for(r)                 # zero-shot: == the cloze
        gen = recall.greedy_generation(model, tokenizer, prompt)
        # (2) one teacher-forced forward -> gold_prob + all internal features.
        feats, hidden, neurons = internal.extract_fact(
            model, tokenizer, r, positions=positions,
            neuron_eps=config.NEURON_EPS, top_k=top_k)
        if dump is not None:
            dump.write(i, hidden)
        if ndump is not None:
            ndump.write(i, neurons)
        rows.append({
            "id": r["id"],
            "step": step,
            "correct": lment_popqa.is_correct(r, gen),
            "excluded": lment_popqa.is_excluded(r),
            "prompt": prompt,
            "generation": gen,
            **feats,
        })
    if dump is not None:
        dump.close()
    if ndump is not None:
        ndump.close()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="*", default=config.CHECKPOINT_STEPS)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N facts (smoke test)")
    ap.add_argument("--min-shared-chunks", type=int, default=0,
                    help="drop facts whose subject and answer never co-occur in a "
                         "shared_chunk (len(shared_chunks) < N); such facts can never "
                         "get a first_learned_step or timeline features since they "
                         "have no exposure to be tracked. Default 0 = keep everything.")
    ap.add_argument("--no-hidden", action="store_true",
                    help="skip the f16 hidden-state dump (scalar features only)")
    ap.add_argument("--positions", nargs="*", default=list(config.HIDDEN_POSITIONS),
                    help="token positions to dump hidden states for")
    ap.add_argument("--top-k", type=int, default=config.NEURON_TOPK,
                    help="also dump the indices of the top-k active MLP neurons "
                         "per (fact, layer) -> neurons_step{N}_*.npy (off by default)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="artifacts root to write into (gets recall/ and internal/ "
                         "subdirs); defaults to config.ARTIFACT_DIR, i.e. "
                         "$LMENT_DATA_ROOT/artifacts (./data/artifacts if unset). "
                         "Use this to avoid surprises from running from the wrong cwd.")
    args = ap.parse_args()

    if args.output_dir:
        out_root = Path(args.output_dir).resolve()
        recall_dir, internal_dir = out_root / "recall", out_root / "internal"
        recall_dir.mkdir(parents=True, exist_ok=True)
        internal_dir.mkdir(parents=True, exist_ok=True)
    else:
        recall_dir, internal_dir = config.RECALL_DIR, config.INTERNAL_DIR
    print(f"[dirs] recall -> {recall_dir}")
    print(f"[dirs] internal -> {internal_dir}")

    df = D.load_popqa_df()
    if args.min_shared_chunks > 0:
        n_before = len(df)
        n_shared = df["shared_chunks"].apply(lambda c: len(c) if c is not None else 0)
        df = df[n_shared >= args.min_shared_chunks].reset_index(drop=True)
        print(f"[filter] dropped {n_before - len(df)} facts with "
              f"< {args.min_shared_chunks} shared_chunks ({len(df)} remain)")
    if args.limit:
        df = df.head(args.limit)
    tokenizer = D.load_tokenizer()

    for step in args.steps:
        out_path = recall_dir / f"recall_step{step}.parquet"
        if out_path.exists():
            print(f"[skip] {out_path.name} exists")
            continue
        print(f"[load] checkpoint step{step}")
        model = D.load_checkpoint(step)
        res = eval_step(model, tokenizer, df, step,
                        dump_hidden=not args.no_hidden, positions=args.positions,
                        top_k=args.top_k, internal_dir=internal_dir)
        res.to_parquet(out_path)
        print(f"[write] {out_path}  ({res['correct'].mean():.3f} correct, "
              f"{res['is_top1'].mean():.3f} top-1)")
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
