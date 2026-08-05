"""Evaluate PopQA recall for a model trained on the (rewritten) corpus.

Recall-only adaptation of the main repo's feature_extraction/eval_checkpoints.py:
one greedy decode per fact -> `correct` via the olmes substring match. The
few-shot machinery (held-out demonstration pool, modes) is identical, so
numbers are directly comparable with the main project's artifacts.

The model can be:
  * a local directory of HF-converted checkpoints (convert_checkpoints.py
    output): --model <EXPLICIT_ROOT>/hf/explicit_170M_1E --steps 10000 ...
  * a published LMEnt repo (baseline):  --model dhgottesman/LMEnt-170M-1E

Usage:
    python eval_explicit.py --model .../hf/explicit_170M_1E --steps 109672 \
        --num-shots 5 --fewshot-mode only-same-prop --run-tag explicit_170M_1E
    python eval_explicit.py --model dhgottesman/LMEnt-170M-1E --steps 109672 \
        --num-shots 5 --fewshot-mode only-same-prop --run-tag baseline_170M_1E
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config
from emlib import data as D
from emlib import fewshot as fewshot_mod
from emlib import lment_popqa, recall
from emlib.fewshot import FewShot


def _checkpoint_rows(rows, partial_path):
    tmp = partial_path.with_suffix(".tmp.parquet")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, partial_path)


def eval_step(model, tokenizer, df, step, *, write_every=0, partial_path=None,
              fewshot=None):
    ids = df["id"].tolist()
    rows, done_ids = [], set()
    if partial_path is not None and partial_path.exists():
        rows = pd.read_parquet(partial_path).to_dict("records")
        done_ids = {r["id"] for r in rows}
        print(f"[resume] step{step}: {len(done_ids)}/{len(ids)} facts already done")

    n_new = 0
    for _, r in df.iterrows():
        if r["id"] in done_ids:
            continue
        prompt = D.prompt_for(r, prefix=fewshot.prefix_for(r) if fewshot else "")
        gen = recall.greedy_generation(model, tokenizer, prompt)
        rows.append({
            "id": r["id"],
            "step": step,
            "num_shots": fewshot.shots_for(r) if fewshot is not None else 0,
            "correct": lment_popqa.is_correct(r, gen),
            "excluded": lment_popqa.is_excluded(r),
            "prompt": prompt,
            "generation": gen,
        })
        n_new += 1
        if write_every and partial_path is not None and n_new % write_every == 0:
            _checkpoint_rows(rows, partial_path)
            print(f"[partial] step{step}: {len(rows)}/{len(ids)} facts", flush=True)

    order = {fid: k for k, fid in enumerate(ids)}
    return (pd.DataFrame(rows)
              .sort_values("id", key=lambda s: s.map(order))
              .reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="local converted-checkpoints dir or an HF repo id")
    ap.add_argument("--run-tag", required=True,
                    help="output name: artifacts/<run-tag>/recall_step{N}*.parquet")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="checkpoint steps to evaluate (default: all local "
                         "steps, or required for HF repos)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-shared-chunks", type=int, default=0)
    ap.add_argument("--write-every", type=int, default=1000)
    ap.add_argument("--num-shots", type=int, default=config.NUM_SHOTS)
    ap.add_argument("--fewshot-mode", choices=fewshot_mod.MODES,
                    default="only-same-prop")
    ap.add_argument("--fewshot-source", choices=("ours", "upstream"), default="ours")
    ap.add_argument("--fewshot-per-prop", type=int, default=None)
    args = ap.parse_args()

    # A path-like --model that is not an existing directory would otherwise be
    # sent to the HF Hub as a repo id, failing with an opaque HFValidationError.
    # The usual cause is a dropped leading "/".
    looks_like_path = ("/" in args.model and not args.model.startswith("hf://")
                       and args.model.count("/") > 1)
    if looks_like_path and not os.path.isdir(args.model):
        hint = ""
        if not os.path.isabs(args.model) and os.path.isdir("/" + args.model):
            hint = f"\n  did you mean the absolute path?  /{args.model}"
        ap.error(f"--model looks like a filesystem path but is not a directory: "
                 f"{args.model}{hint}\n"
                 f"  (a real HF repo id has exactly one '/', e.g. "
                 f"dhgottesman/LMEnt-170M-1E)")

    D.set_model_spec(args.model)
    steps = args.steps
    if steps is None:
        if os.path.isdir(args.model):
            steps = D._available_steps(args.model)
        else:
            ap.error("--steps is required when --model is an HF repo id")
    out_dir = config.ARTIFACTS_DIR / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[model] {args.model}\n[steps] {steps}\n[out] {out_dir}")

    df = D.load_popqa_df()
    fewshot = FewShot.build(df, num_shots=args.num_shots,
                            source=args.fewshot_source,
                            per_prop=args.fewshot_per_prop,
                            mode=args.fewshot_mode)
    df = fewshot.drop_subjects(df)
    if args.min_shared_chunks > 0:
        n_shared = df["shared_chunks"].apply(lambda c: len(c) if c is not None else 0)
        df = df[n_shared >= args.min_shared_chunks].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    tokenizer = D.load_tokenizer()

    tag = "" if args.num_shots == 0 else f"_{args.num_shots}shot"
    if args.num_shots > 0 and args.fewshot_mode == "fixed-block":
        tag += "_sameprop"
    elif args.num_shots > 0 and args.fewshot_mode == "only-same-prop":
        tag += "_onlysameprop"
    tag += "_recallonly"

    for step in steps:
        out_path = out_dir / f"recall_step{step}{tag}.parquet"
        if out_path.exists():
            print(f"[skip] {out_path.name} exists")
            continue
        partial_path = out_dir / f"recall_step{step}{tag}.partial.parquet"
        print(f"[load] checkpoint step{step}")
        model = D.load_checkpoint(step)
        res = eval_step(model, tokenizer, df, step,
                        write_every=args.write_every, partial_path=partial_path,
                        fewshot=fewshot)
        res.to_parquet(out_path)
        partial_path.unlink(missing_ok=True)
        print(f"[write] {out_path}  ({res['correct'].mean():.3f} correct)")
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
