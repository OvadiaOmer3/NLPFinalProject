"""Step A.1 - Acquisition timeline (+ internal states).

Sweep every checkpoint and, for every fact, record:
  * recall (substring match) + gold-answer probability   -> the acquisition curve
  * internal-state features from a single forward pass    -> how the fact is stored
  * (optional) the full per-layer hidden states in f16    -> probes / drift offline

Writes one enriched parquet per checkpoint to ``recall/`` and, unless
``--no-hidden`` is passed, one f16 hidden-state dump per checkpoint to
``internal/``.  The recall parquet keeps its original columns (so
``build_feature_table.py`` is unaffected) and gains the internal-feature columns.

Cost model: we touch the model exactly twice per (checkpoint, fact) -- once for
the greedy decode that ``correct`` needs, once for the teacher-forced forward
that yields *everything else* (gold_prob, logit-lens depth, next-token geometry,
per-layer residual / MLP activity, and the hidden-state dump).

Run once per checkpoint (ideally a Slurm array job, one GPU each) or loop locally.

Usage:
    python feature_extraction/eval_checkpoints.py                 # all checkpoints
    python feature_extraction/eval_checkpoints.py --steps 0 10000 # a subset
    python feature_extraction/eval_checkpoints.py --model dhgottesman/LMEnt-1B-1E
                                                                  # another release; the
                                                                  # checkpoint grid follows
                                                                  # (pair with --output-dir,
                                                                  # filenames don't carry the
                                                                  # model)
    python feature_extraction/eval_checkpoints.py --model data/hf/explicit_170M_1E \
        --run-tag translated_corpus/explicit_170M_1E --no-internal \
        --num-shots 5 --fewshot-mode only-same-prop                # a LOCAL dir of converted
                                                                  # checkpoints (explicit_mentions/
                                                                  # convert_checkpoints.py); steps
                                                                  # are read off the disk, output
                                                                  # goes to artifacts/recall/<run-tag>/
    python feature_extraction/eval_checkpoints.py --no-hidden     # scalars only, no dump
    python feature_extraction/eval_checkpoints.py --no-internal   # recall only: correct +
                                                                  # generation, one pass/fact
    python feature_extraction/eval_checkpoints.py --limit 100     # smoke test on 100 facts
    python feature_extraction/eval_checkpoints.py --prompts P.parquet
                                                                  # use pre-built prompts
                                                                  # (feature_extraction/
                                                                  # build_prompts.py) instead of
                                                                  # rendering them from the flags
                                                                  # below -- the way to keep the
                                                                  # prompts constant while
                                                                  # comparing models
    python feature_extraction/eval_checkpoints.py --num-shots 0   # bare zero-shot cloze
    python feature_extraction/eval_checkpoints.py --fewshot-mode only-same-prop
                                                                  # demos of the query's OWN
                                                                  # relation, none sharing its
                                                                  # answer (see src/fewshot.py)
    python feature_extraction/eval_checkpoints.py --fewshot-source upstream
                                                                  # LMEnt's verbatim demos
                                                                  # (leaky; for parity only)
    python feature_extraction/eval_checkpoints.py --output-dir D:/lment_out  # explicit output root
                                                                  # (cwd-independent; overrides
                                                                  # $LMENT_DATA_ROOT)
    python feature_extraction/eval_checkpoints.py --min-shared-chunks 1  # drop facts that can
                                                                  # never be "seen" (see below)
    python feature_extraction/eval_checkpoints.py --top-shared-chunks-pct 10
                                                                  # keep only the top decile of
                                                                  # facts by num_shared_chunks
    python feature_extraction/eval_checkpoints.py --num-shots 15 --internal-fewshot
                                                                  # measure the internals on the
                                                                  # 15-shot prompt too, not on the
                                                                  # bare zero-shot cloze
    python feature_extraction/eval_checkpoints.py --top-k 128      # also dump the top-k active
                                                                  # MLP neurons per (fact, layer):
                                                                  # indices + signed activations
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src import data as D
from src import lment_popqa
from src import recall
from src.features import internal
from src import fewshot as fewshot_mod
from src.fewshot import FewShot
from src.prompt_set import PromptSet, fewshot_tag


def _checkpoint_rows(rows, partial_path, dump, ndump):
    """Persist progress so far: flush the memmaps, then swap in a new partial
    parquet.  Written to a temp file and ``os.replace``d so a kill mid-write
    leaves the previous partial intact rather than a truncated file."""
    if dump is not None:
        dump.flush()
    if ndump is not None:
        ndump.flush()          # flushes the value memmaps too, not just the indices
    tmp = partial_path.with_suffix(".tmp.parquet")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, partial_path)


def eval_step(model, tokenizer, df, step, *, dump_hidden, positions, top_k=None,
             internal_dir=None, write_every=0, partial_path=None, prompts=None,
             internal_feats=True, neuron_values=True, internal_fewshot=False):
    """Full single-checkpoint pass -> a DataFrame (one row per fact).

    ``prompts`` is a pre-rendered ``src.prompt_set.PromptSet`` (the bare cloze
    when None).  Also streams the f16 hidden-state dump (when ``dump_hidden``)
    and the top-k active-neuron dump (when ``top_k``) to ``internal_dir``
    (defaults to ``config.INTERNAL_DIR``, i.e. ``artifacts/internal/``).  The
    neuron dump writes indices and, unless ``neuron_values=False``, their signed
    activations.

    With ``write_every > 0``, a partial parquet is written to ``partial_path``
    every N facts.  If that file already exists on entry its facts are skipped
    and the dumps are reopened in place, so a crashed or preempted run resumes
    instead of redoing the whole checkpoint.
    """
    internal_dir = internal_dir or config.INTERNAL_DIR
    # Without the internal features there is no second forward pass at all, so
    # neither dump has anything to write.
    if not internal_feats:
        dump_hidden, top_k = False, None
    ids = df["id"].tolist()

    rows, done_ids = [], set()
    if partial_path is not None and partial_path.exists():
        rows = pd.read_parquet(partial_path).to_dict("records")
        done_ids = {r["id"] for r in rows}
        print(f"[resume] step{step}: {len(done_ids)}/{len(ids)} facts already done")

    dump = None
    if dump_hidden:
        dump = internal.HiddenDump(
            step, ids, model.config.num_hidden_layers + 1,
            model.config.hidden_size, positions, internal_dir,
            resume=bool(done_ids))
    ndump = None
    if top_k:
        # neurons are captured at answer + subject_last (both slots the hook sees).
        ndump = internal.NeuronDump(
            step, ids, model.config.num_hidden_layers, top_k,
            ("answer", "subject_last"), internal_dir, resume=bool(done_ids),
            store_values=neuron_values)

    n_new = 0
    for i, (_, r) in enumerate(df.iterrows()):
        if r["id"] in done_ids:
            continue                           
        # (1) greedy decode -> substring-match correctness (paper's `correct`).
        # Prompts are pre-rendered (src/prompt_set.py) so every checkpoint and every
        # model scores the identical strings; `prompts=None` is the bare cloze.
        prompt = prompts.prompt(r["id"]) if prompts is not None else D.prompt_for(r)
        gen = recall.greedy_generation(model, tokenizer, prompt)
        # (2) one teacher-forced forward -> gold_prob + all internal features.
        # Skipped entirely with --no-internal: that forward asks for every layer's
        # hidden state, hooks every mlp.down_proj, and runs the LM head once per
        # layer for the logit lens.  None of it is needed for correct/generation.
        # The teacher-forced pass reads the ZERO-SHOT cloze by default, so the
        # answer position stays unconditioned and comparable across shot counts
        # even when `correct` above was scored on a 15-shot prompt.  With
        # --internal-fewshot it reads the same prompt the decode saw instead.
        if internal_feats:
            feats, hidden, neurons = internal.extract_fact(
                model, tokenizer, r, positions=positions,
                neuron_eps=config.NEURON_EPS, top_k=top_k,
                prompt=prompt if internal_fewshot else None)
            if dump is not None:
                dump.write(i, hidden)
            if ndump is not None:
                ndump.write(i, neurons)
        else:
            feats = {}
        rows.append({
            "id": r["id"],
            "step": step,
            # The count this fact actually got, not the requested one: under
            # --fewshot-mode only-same-prop a fact whose answer knocks out part of
            # its relation's bench can end up with fewer.
            "num_shots": prompts.shots(r["id"]) if prompts is not None else 0,
            "correct": lment_popqa.is_correct(r, gen),
            "excluded": lment_popqa.is_excluded(r),
            "prompt": prompt,
            "generation": gen,
            **feats,
        })
        n_new += 1
        if write_every and partial_path is not None and n_new % write_every == 0:
            _checkpoint_rows(rows, partial_path, dump, ndump)
            print(f"[partial] step{step}: {len(rows)}/{len(ids)} facts", flush=True)
    if dump is not None:
        dump.close()
    if ndump is not None:
        ndump.close()
    # Resumed rows come first; restore the dataframe's fact order so the parquet
    # lines up with the dumps' ids.npy.
    order = {fid: k for k, fid in enumerate(ids)}
    return (pd.DataFrame(rows)
              .sort_values("id", key=lambda s: s.map(order))
              .reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=config.MODEL_ID,
                    help=f"HuggingFace repo of the LMEnt release to evaluate "
                         f"(default {config.MODEL_ID}, or $LMENT_MODEL_ID). Known "
                         f"releases: {', '.join(config.MODEL_TOTAL_STEPS)}. Or a LOCAL "
                         f"directory of converted checkpoints (<dir>/stepN/, from "
                         f"explicit_mentions/convert_checkpoints.py), whose steps are "
                         f"discovered on disk. Artifact filenames do NOT encode the "
                         f"model, so give each one its own --run-tag / --output-dir.")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="checkpoint steps to evaluate; defaults to every "
                         "checkpoint the --model publishes (step0 .. final, every "
                         f"{config.CHECKPOINT_INTERVAL}) or, for a local dir, every "
                         "stepN/ it contains.")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N facts (smoke test)")
    ap.add_argument("--min-shared-chunks", type=int, default=0,
                    help="drop facts whose subject and answer never co-occur in a "
                         "shared_chunk (len(shared_chunks) < N); such facts can never "
                         "get a first_learned_step or timeline features since they "
                         "have no exposure to be tracked. Default 0 = keep everything. "
                         "Mutually exclusive with --top-shared-chunks-pct.")
    ap.add_argument("--top-shared-chunks-pct", type=float, default=None,
                    help="keep only the top N%% of facts by num_shared_chunks (e.g. 10 "
                         "for the top decile -- the facts with the most exposure to "
                         "learn from). The cutoff is a quantile of the FULL dataset "
                         "(before --limit / fewshot subject-drop), so it stays the same "
                         "count regardless of --fewshot-mode or --num-shots, and is "
                         "resolved to an equivalent --min-shared-chunks. Mutually "
                         "exclusive with --min-shared-chunks.")
    ap.add_argument("--no-hidden", action="store_true",
                    help="skip WRITING the f16 hidden-state dump. Note the states "
                         "are still computed -- use --no-internal to skip the work.")
    ap.add_argument("--no-internal", action="store_true",
                    help="skip the internal-feature forward pass entirely: output "
                         "keeps only id/step/num_shots/correct/excluded/prompt/"
                         "generation. Drops gold_prob, logit-lens depth, next-token "
                         "geometry, per-layer MLP activity and both dumps. Use this "
                         "when you only care about recall -- it halves the model "
                         "calls per fact and avoids running the LM head once per "
                         "layer. Note build_feature_table.py needs the dropped "
                         "columns, so such runs are for recall inspection only.")
    ap.add_argument("--internal-fewshot", action="store_true",
                    help="measure the internal features on the SAME few-shot prompt "
                         "the greedy decode saw, instead of on the bare zero-shot "
                         "cloze. Default (off) keeps the answer position "
                         "unconditioned, so the lens/hidden states are comparable "
                         "across shot counts and with every artifact produced so far "
                         "-- at the cost of `correct` and the internals living on two "
                         "different prompts. Turning it on makes them agree but "
                         "changes what the depth profile describes: retrieval under "
                         "in-context conditioning rather than how the fact is stored. "
                         "No-op at --num-shots 0; conflicts with --no-internal. "
                         "Output is tagged `_fsinternal` so it never collides with "
                         "the zero-shot-internals parquets.")
    ap.add_argument("--write-every", type=int, default=1000,
                    help="checkpoint progress to recall_step{N}.partial.parquet "
                         "every N facts; a rerun resumes from it instead of "
                         "redoing the step. 0 = only write at the end.")
    ap.add_argument("--positions", nargs="*", default=list(config.HIDDEN_POSITIONS),
                    help="token positions to dump hidden states for")
    ap.add_argument("--top-k", type=int, default=config.NEURON_TOPK,
                    help="also dump the top-k active MLP neurons per (fact, layer) "
                         "-> neurons_step{N}_{pos}_idx.npy (uint16 indices) and "
                         "_val.npy (float16 signed activations). Off by default. "
                         "k=128 is the recommended setting: the measured "
                         "participation ratio (effective neuron count) is p99.9 "
                         "~58 at both positions, so 128 gives 2x headroom, and the "
                         "whole dump is still only ~6%% of the hidden-state dump. "
                         "k is a truncation you cannot undo -- top-32 can be "
                         "derived from a top-128 dump, not the reverse.")
    ap.add_argument("--no-neuron-values", action="store_true",
                    help="with --top-k, dump only the neuron indices and skip the "
                         "signed-activation values (halves the neuron dump). "
                         "Set-overlap analyses still work; magnitude-weighted "
                         "overlap and knowledge-entropy measures do not.")
    ap.add_argument("--prompts", type=str, default=None,
                    help="parquet of pre-built prompts (feature_extraction/"
                         "build_prompts.py). It fixes the prompt strings, the fact "
                         "set and its order, so different models/checkpoints are "
                         "compared on identical inputs even if the selection rule "
                         "or the dataset later changes. Incompatible with the "
                         "--num-shots / --fewshot-* / --min-shared-chunks / "
                         "--top-shared-chunks-pct flags -- those were settled when "
                         "the file was built. Without it the prompts are rendered "
                         "from those flags, as before.")
    # default=None (not config.NUM_SHOTS) so the --prompts conflict check below can
    # tell "not passed" from "passed"; it is normalised to config.NUM_SHOTS after.
    ap.add_argument("--num-shots", type=int, default=None,
                    help="in-context demonstrations prepended to the cloze "
                         f"(default {config.NUM_SHOTS}). 0 = bare zero-shot cloze. "
                         "Demonstrations come from a held-out pool, never from the "
                         "eval set -- see src/fewshot.py. The pool's subjects are "
                         "dropped from the eval set at EVERY shot count, so 0-shot "
                         "and n-shot score the identical fact set.")
    ap.add_argument("--fewshot-mode", choices=fewshot_mod.MODES, default=None,
                    help="which demonstrations a fact is shown -- the knob for "
                         "comparing few-shot methods. 'drop-same-prop' (default): "
                         "the pool minus the query's own relation, so the answer "
                         "type is never handed over (15 of a 16-fact pool). "
                         "'fixed-block': one constant block including the query's "
                         "relation (upstream's behaviour; pair with --num-shots 16 "
                         "to use the whole pool). 'only-same-prop': every demo "
                         "shares the query's relation and none has an answer that "
                         "would be scored as the query's answer -- the answer type "
                         "for free, copying impossible. That mode selects its own "
                         "larger pool (num-shots+5 facts per relation by default) "
                         "and therefore holds out ~300 eval facts instead of 16, so "
                         "compare it against other modes on the id intersection.")
    ap.add_argument("--fewshot-keep-same-prop", action="store_true",
                    help="deprecated alias for --fewshot-mode fixed-block.")
    ap.add_argument("--fewshot-source", choices=("ours", "upstream"), default="ours",
                    help="'ours' selects demonstrations by the rule in "
                         "src/fewshot.py. 'upstream' uses the fork's verbatim "
                         'FEWSHOT_SOURCES["popqa"] -- for reproducing LMEnt\'s '
                         "numbers as-is, including its answer-in-question demo "
                         "and its frequent-answer demos.")
    ap.add_argument("--fewshot-per-prop", type=int, default=None,
                    help="demonstrations to select per prop_id. Only affects "
                         "--fewshot-source ours. Default depends on the mode: 1 "
                         "(-> a 16-fact pool, one per relation), or --num-shots + "
                         f"{fewshot_mod.SAME_PROP_MARGIN} under --fewshot-mode "
                         "only-same-prop, which shows a fact its own relation only "
                         "and so needs a bench per relation.")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="artifacts root to write into (gets recall/ and internal/ "
                         "subdirs); defaults to config.RECALL_DIR and "
                         "config.INTERNAL_DIR, i.e. $LMENT_DATA_ROOT/artifacts/{recall,internal} "
                         "(NLPFinalProject/data/... if unset). "
                         "Use this to avoid surprises from running from the wrong cwd.")
    ap.add_argument("--run-tag", type=str, default=None,
                    help="subdirectory under config.RECALL_DIR (and INTERNAL_DIR) to "
                         "write into, e.g. translated_corpus/explicit_170M_1E; slashes "
                         "allowed. One per model, since filenames do not carry the "
                         "model. Mutually exclusive with --output-dir.")
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

    # Before any load: everything downstream reads config.MODEL_ID at call time.
    # For a local checkpoint dir this also sets the grid to its stepN/ dirs.
    config.use_model(args.model)
    if args.steps is None:
        args.steps = config.CHECKPOINT_STEPS
        if args.steps is None:
            ap.error(f"{args.model} is not in config.MODEL_TOTAL_STEPS and is not a "
                     f"local checkpoint directory, so its checkpoint grid is unknown "
                     f"-- pass --steps explicitly (or add the release to the table).")
    print(f"[model] {config.MODEL_ID} | {len(args.steps)} step(s), "
          f"final published step{config.FINAL_STEP}")

    # --fewshot-keep-same-prop predates --fewshot-mode and means fixed-block; keep it
    # working, but refuse to guess when the two disagree.
    if args.fewshot_keep_same_prop:
        if args.fewshot_mode not in (None, "fixed-block"):
            ap.error(f"--fewshot-keep-same-prop conflicts with --fewshot-mode "
                     f"{args.fewshot_mode} (it is an alias for fixed-block)")
        args.fewshot_mode = "fixed-block"

    # A prompt file already answers every prompt-shaping question, so taking one of
    # these flags too would mean silently honouring one and ignoring the other.
    if args.prompts:
        conflicting = [name for name, val in (
            ("--num-shots", args.num_shots),
            ("--fewshot-mode", args.fewshot_mode),
            ("--fewshot-per-prop", args.fewshot_per_prop),
            ("--min-shared-chunks", args.min_shared_chunks or None),
            ("--top-shared-chunks-pct", args.top_shared_chunks_pct),
        ) if val is not None] + (
            ["--fewshot-source"] if args.fewshot_source != "ours" else [])
        if conflicting:
            ap.error(f"--prompts is incompatible with {', '.join(conflicting)}: the "
                     "prompt file was built with its own settings (inspect them with "
                     "build_prompts.py --inspect). Rebuild the file to change them.")
    args.fewshot_mode = args.fewshot_mode or "drop-same-prop"
    args.num_shots = config.NUM_SHOTS if args.num_shots is None else args.num_shots

    if args.top_shared_chunks_pct is not None:
        if args.min_shared_chunks > 0:
            ap.error("--top-shared-chunks-pct and --min-shared-chunks are mutually "
                     "exclusive")
        if not (0 < args.top_shared_chunks_pct <= 100):
            ap.error("--top-shared-chunks-pct must be in (0, 100]")

    if args.internal_fewshot:
        if args.no_internal:
            ap.error("--internal-fewshot needs the internal forward pass that "
                     "--no-internal skips")
        if args.num_shots == 0:
            ap.error("--internal-fewshot is a no-op at --num-shots 0: the few-shot "
                     "prompt IS the zero-shot cloze there")

    if args.output_dir and args.run_tag:
        ap.error("--output-dir and --run-tag are mutually exclusive")
    if args.output_dir:
        out_root = Path(args.output_dir).resolve()
        recall_dir, internal_dir = out_root / "recall", out_root / "internal"
        recall_dir.mkdir(parents=True, exist_ok=True)
        internal_dir.mkdir(parents=True, exist_ok=True)
    elif args.run_tag:
        recall_dir = config.RECALL_DIR / args.run_tag
        internal_dir = config.INTERNAL_DIR / args.run_tag
        recall_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_internal:
            internal_dir.mkdir(parents=True, exist_ok=True)
    else:
        recall_dir, internal_dir = config.RECALL_DIR, config.INTERNAL_DIR
    print(f"[dirs] recall -> {recall_dir}")
    print(f"[dirs] internal -> {internal_dir}")

    df = D.load_popqa_df()
    if args.top_shared_chunks_pct is not None:
        # Quantile of the FULL dataset, before any subject-drop / --limit, so the
        # resulting count threshold means the same thing (and selects the same
        # facts) no matter which --fewshot-mode or --num-shots is in play.
        # (Rejected above under --prompts: the file already fixed the fact set.)
        n_shared_full = df["shared_chunks"].apply(lambda c: len(c) if c is not None else 0)
        args.min_shared_chunks = int(n_shared_full.quantile(1 - args.top_shared_chunks_pct / 100))
        print(f"[filter] --top-shared-chunks-pct {args.top_shared_chunks_pct:g} -> "
              f"--min-shared-chunks {args.min_shared_chunks} "
              f"({(n_shared_full >= args.min_shared_chunks).sum()}/{len(df)} facts, "
              "full dataset)")
    if args.prompts:
        # The file IS the eval set: the demonstration pool's subjects are already
        # gone and its filters already applied, so `df` is only reordered to match.
        prompts = PromptSet.load(args.prompts)
        print(f"[prompts] {args.prompts}")
        print(prompts.describe())
        if args.limit:
            prompts = prompts.head(args.limit)
        df = prompts.select(df)
    else:
        # Demonstrations are chosen from the FULL dataset, before the filters below,
        # so the pool doesn't shift when --limit / --min-shared-chunks change.  Their
        # subjects then leave the eval set at every shot count (0 included) so runs
        # remain comparable.
        fewshot = FewShot.build(df, num_shots=args.num_shots,
                                source=args.fewshot_source,
                                per_prop=args.fewshot_per_prop,
                                mode=args.fewshot_mode)
        df = fewshot.drop_subjects(df)
        if args.min_shared_chunks > 0:
            n_before = len(df)
            n_shared = df["shared_chunks"].apply(lambda c: len(c) if c is not None else 0)
            df = df[n_shared >= args.min_shared_chunks].reset_index(drop=True)
            print(f"[filter] dropped {n_before - len(df)} facts with "
                  f"< {args.min_shared_chunks} shared_chunks ({len(df)} remain)")
        if args.limit:
            df = df.head(args.limit)
        # Rendered up front rather than inside the loop, so the demo selection is
        # paid for once instead of once per checkpoint.
        prompts = PromptSet.build(df, fewshot, source=args.fewshot_source)
    tokenizer = D.load_tokenizer()

    # `correct` / `generation` / `prompt` depend on the shot count, so the filename
    # must too -- otherwise the `exists -> skip` guard below would hand a 0-shot
    # run the 15-shot parquet.  0-shot keeps the original bare name so existing
    # artifacts and downstream globs are untouched.  The mode and the source change
    # the prompt too, so they also reach the filename (fixed-block keeps the
    # "_sameprop" suffix its artifacts already carry).  With --prompts the tag comes
    # from the file's metadata, so the same settings land on the same filenames
    # whichever way the prompts were produced.
    tag = (prompts.tag if args.prompts else
           fewshot_tag(args.num_shots, args.fewshot_mode, args.fewshot_source))
    # A --no-internal parquet has strictly fewer columns, so it must not land on
    # the filename a full run uses -- otherwise the `exists -> skip` guard would
    # feed build_feature_table.py a table with no gold_prob / internal features.
    if args.no_internal:
        tag += "_recallonly"
    # Same population and same shot count, but a different measured prompt, so the
    # internal columns are not comparable with a default run's -- it needs its own
    # filename or the `exists -> skip` guard would serve the zero-shot-internals
    # parquet in its place.
    if args.internal_fewshot:
        tag += "_fsinternal"
    # The population changes with the filter, so it has to reach the filename too --
    # otherwise the `exists -> skip` guard would hand a filtered run the unfiltered
    # parquet (or vice versa).
    if args.top_shared_chunks_pct is not None:
        tag += f"_top{args.top_shared_chunks_pct:g}pct"
    elif args.min_shared_chunks > 0:
        tag += f"_minshared{args.min_shared_chunks}"

    for step in args.steps:
        out_path = recall_dir / f"recall_step{step}{tag}.parquet"
        if out_path.exists():
            print(f"[skip] {out_path.name} exists")
            continue
        partial_path = recall_dir / f"recall_step{step}{tag}.partial.parquet"
        print(f"[load] checkpoint step{step}")
        model = D.load_checkpoint(step)
        res = eval_step(model, tokenizer, df, step,
                        dump_hidden=not args.no_hidden, positions=args.positions,
                        top_k=args.top_k, internal_dir=internal_dir,
                        write_every=args.write_every, partial_path=partial_path,
                        prompts=prompts, internal_feats=not args.no_internal,
                        neuron_values=not args.no_neuron_values,
                        internal_fewshot=args.internal_fewshot)
        res.to_parquet(out_path)
        partial_path.unlink(missing_ok=True)
        msg = f"[write] {out_path}  ({res['correct'].mean():.3f} correct"
        if "is_top1" in res.columns:            # absent under --no-internal
            msg += f", {res['is_top1'].mean():.3f} top-1"
        print(msg + ")")
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
