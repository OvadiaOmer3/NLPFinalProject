"""Generate an OLMo-core training config for a corpus built by
build_explicit_corpus.py (or for the original corpus, as a control run).

Examples:
    # Train the compute-optimal 170M for 1 epoch on the fully explicit corpus
    python make_train_config.py --run-name explicit_170M_1E --corpus explicit_all

    # Control: identical run on the original corpus
    python make_train_config.py --run-name original_170M_1E --corpus original

    # Two-phase: 1 epoch original, then 1 epoch explicit starting from phase 1
    python make_train_config.py --run-name phase1_170M --corpus original
    python make_train_config.py --run-name phase2_170M --corpus explicit_all \
        --load-path <EXPLICIT_ROOT>/runs/phase1_170M/olmo2_170M_.../stepNNNN

Prints the torchrun command to launch (see also slurm/train.sbatch).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True,
                    help="run directory name under runs/")
    ap.add_argument("--corpus", required=True,
                    help="a corpus tag under corpora/ (e.g. explicit_all), "
                         "'original' for the authors' corpus, or an absolute "
                         "path to a corpus root containing dataset-tokenized/")
    ap.add_argument("--model", default="olmo2_170M",
                    choices=["olmo2_170M", "olmo2_190M", "olmo2_600M", "olmo2_1B"])
    ap.add_argument("--epochs", type=int, default=1)
    # Paper S B.4: "All LMEnt models were trained using the AdamW optimizer
    # with a global batch size of 32,768, rank batch size of 8,192, peak
    # learning rate of 5e-3, weight decay of 0.05, and 1,000 warmup steps."
    ap.add_argument("--lr", type=float, default=5e-3,
                    help="peak LR (paper S B.4)")
    ap.add_argument("--weight-decay", type=float, default=0.05,
                    help="weight decay (paper S B.4)")
    ap.add_argument("--global-batch-size", type=int, default=32768)
    ap.add_argument("--rank-microbatch-size", type=int, default=8192)
    ap.add_argument("--gpus", type=int, default=4,
                    help="how many GPUs the training job will use. OLMo-core "
                         "requires global_batch_size %% (rank_microbatch_size * "
                         "gpus) == 0, so this is validated here rather than "
                         "failing at startup. With the paper's 32768/8192 that "
                         "allows 1, 2 or 4 GPUs; for 8 use "
                         "--rank-microbatch-size 4096.")
    ap.add_argument("--metrics-interval", type=int, default=100,
                    help="steps between console log lines AND metric "
                         "collection. Each collection does a cross-rank "
                         "reduction plus a host-device sync, and the console "
                         "otherwise prints one line per step (~114k lines, "
                         "~82 MB). W&B still receives per-step points either "
                         "way, since metrics are buffered per step and flushed "
                         "in batches.")
    ap.add_argument("--warmup-steps", type=int, default=1000,
                    help="LR warmup steps (paper S B.4)")
    ap.add_argument("--save-interval", type=int, default=None,
                    help="steps between PERMANENT checkpoints. Default: none -- "
                         "only the final model is kept (plus one rolling "
                         "ephemeral checkpoint for crash/preemption resume). "
                         "Set e.g. 10000 to also keep intermediates for a "
                         "learning-dynamics analysis.")
    ap.add_argument("--ephemeral-save-interval", type=int, default=1000,
                    help="steps between rolling resume checkpoints; only the "
                         "most recent one is kept on disk")
    ap.add_argument("--load-path", default=None,
                    help="OLMo-core checkpoint dir (…/stepN) to initialize the "
                         "model from -- enables the two-phase schedule")
    ap.add_argument("--wandb", action="store_true",
                    help="log to Weights & Biases (needs WANDB_API_KEY in the "
                         "job environment -- see README)")
    ap.add_argument("--wandb-project", default="lment-explicit-mentions",
                    help="W&B project name")
    ap.add_argument("--wandb-entity", default=None,
                    help="W&B entity (your username or team). Defaults to the "
                         "default entity of the logged-in account.")
    ap.add_argument("--wandb-group", default=None,
                    help="W&B group, for keeping explicit/control runs together")
    ap.add_argument("--downstream-eval", action="store_true",
                    help="enable OLMo-core's in-training downstream evaluator "
                         "(needs its eval deps; off by default)")
    ap.add_argument("--instance-metadata", action="store_true",
                    help="attach each chunk's entity annotations to its batch. "
                         "OFF by default because it is very expensive and "
                         "nothing in the training path reads it: the KAS "
                         "dataset seeks a random row in dataset-cache/"
                         "dataset-metadata/part-*.csv (~245 GB) for EVERY "
                         "instance, which made data loading 87%% of step time "
                         "on a 170M run. The collator puts the result in the "
                         "batch and only olmo_core.data.kas_metrics would "
                         "consume it -- and nothing imports that module. Turn "
                         "this on only if you add a callback that needs "
                         "batch['metadata']. Does not affect the dataset "
                         "fingerprint, so toggling it reuses the same cache.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # OLMo-core's Trainer asserts this; catch it now instead of after a queue wait.
    if args.global_batch_size % (args.rank_microbatch_size * args.gpus) != 0:
        ok = [n for n in (1, 2, 4, 8, 16)
              if args.global_batch_size % (args.rank_microbatch_size * n) == 0]
        alt = args.global_batch_size // args.gpus
        ap.error(
            f"global_batch_size ({args.global_batch_size}) must be divisible by "
            f"rank_microbatch_size ({args.rank_microbatch_size}) x gpus "
            f"({args.gpus}).\n"
            f"  valid GPU counts at this microbatch size: {ok}\n"
            f"  or keep {args.gpus} GPUs and pass "
            f"--rank-microbatch-size {alt}")
    print(f"[batch] {args.gpus} GPUs x {args.rank_microbatch_size} tokens = "
          f"{args.gpus * args.rank_microbatch_size} tokens per pass; "
          f"{args.global_batch_size // (args.rank_microbatch_size * args.gpus)} "
          f"accumulation step(s) -> {args.global_batch_size} tokens per optimizer step")

    if args.corpus == "original":
        corpus_root = config.SRC_ROOT
    elif Path(args.corpus).is_absolute():
        corpus_root = Path(args.corpus)
    else:
        corpus_root = config.corpus_dir(args.corpus)
    paths_glob = str(corpus_root / "dataset-tokenized" / "*.npy")
    work_dir = str(corpus_root / "dataset-cache")
    if args.corpus == "original":
        # Never write caches into the authors' directory: their dataset-cache
        # is already fully built, so OLMo-core will only read from it.
        print(f"note: using the authors' prebuilt dataset-cache at {work_dir}")

    run_dir = config.RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model": args.model,
        "optim": {"lr": args.lr, "weight_decay": args.weight_decay},
        "dataset": {
            "name": "kas_vsl",
            "max_sequence_length": 2048,
            "min_sequence_length": 64,
            "vsl_curriculum": {"name": "grow_p2", "num_cycles": 8, "balanced": False},
            "paths": [paths_glob],
            "include_instance_metadata": bool(args.instance_metadata),
            "work_dir": work_dir,
        },
        "data_loader": {
            "global_batch_size": args.global_batch_size,
            "seed": args.seed,
            "num_workers": 4,
            "prefetch_factor": 16,
        },
        "trainer": {
            "save_folder": str(run_dir),
            "rank_microbatch_size": args.rank_microbatch_size,
            "save_overwrite": True,
            "max_duration": {"value": args.epochs, "unit": "epochs"},
            "cancel_check_interval": 5,
            "metrics_collect_interval": args.metrics_interval,
            "callbacks": {
                "console_logger": {"log_interval": args.metrics_interval},
                "lr_scheduler": {"warmup_steps": args.warmup_steps},
                "grad_clipper": {"max_grad_norm": 1.0},
                "checkpointer": {
                    # No --save-interval => park it beyond the end of training so
                    # no intermediate permanent checkpoint is ever written. The
                    # trainer's post_train hook still saves the FINAL model, and
                    # the ephemeral checkpoint (one, rolling) covers preemption.
                    "save_interval": args.save_interval or 100_000_000,
                    "ephemeral_save_interval": args.ephemeral_save_interval,
                    "save_async": True,
                    # Skip the step0 checkpoint a fresh run would otherwise write.
                    "pre_train_checkpoint": False,
                },
                "wandb": {
                    "name": args.run_name,
                    "project": args.wandb_project,
                    "entity": args.wandb_entity,
                    "group": args.wandb_group,
                    "tags": [args.model, args.corpus],
                    "cancel_check_interval": 10,
                    "enabled": bool(args.wandb),
                },
                "downstream_evaluator": {
                    "tasks": ["arc_easy", "sciq", "hellaswag", "piqa"],
                    "eval_interval": 10000,
                    "enabled": bool(args.downstream_eval),
                },
            },
        },
        "init_seed": 12536,
    }
    if args.load_path:
        cfg["load_path"] = args.load_path

    out = run_dir / "config.json"
    out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}")
    here = Path(__file__).resolve().parent
    print("\n1) build the dataset cache once per corpus (CPU, hours):")
    print(f"   sbatch slurm/prebuild_cache.sbatch {out}")
    print("\n2) then train (GPU):")
    print(f"   sbatch slurm/train.sbatch {out}")
    print(f"\n   (or directly: torchrun --nproc-per-node=1 "
          f"{here}/train_explicit.py {out})")


if __name__ == "__main__":
    main()
