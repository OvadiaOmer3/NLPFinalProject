"""Central configuration.

Everything that we *inferred* about the LMEnt release (paths, schema names,
training hyper-parameters) is isolated here so it can be verified against the
real download in one place.  Anything tagged ``# VERIFY`` should be double
checked against the actual files before trusting the numbers downstream.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# HuggingFace identifiers
# --------------------------------------------------------------------------- #
# Which LMEnt release we evaluate.  Pick it per run with `--model`
# (feature_extraction/eval_checkpoints.py) or process-wide with $LMENT_MODEL_ID;
# the env var is the safer of the two because it applies before any import (see
# `use_model` below).
MODEL_ID = os.environ.get("LMENT_MODEL_ID", "dhgottesman/LMEnt-1B-6E")  # 1B, 6 epochs
POPQA_DATASET_ID = "dhgottesman/popqa-kas"    # entity-annotated PopQA
LMENT_DATASET_ID = "dhgottesman/LMEnt-Dataset"  # corpus + batch indices + mentions

# --------------------------------------------------------------------------- #
# Checkpoints  (subfolders inside MODEL_ID)
# --------------------------------------------------------------------------- #
# Every release stores a checkpoint every 10k steps plus the final, uneven one:
# step0, step10000, ..., step{total}.  The totals differ per release, so the grid
# MUST follow MODEL_ID: asking LMEnt-1B-6E for step109672 (the *1E* final) 404s
# the subfolder, and transformers then falls back to guessing the model type from
# the repo name -- "dhgottesman" contains "esm" -- and dies with an unrelated
# "EsmConfig is not supported by AutoModelForCausalLM".  Hence the guard in
# src/data.py:load_checkpoint.  Verified against the repo trees on the Hub.
CHECKPOINT_INTERVAL = 10000
MODEL_TOTAL_STEPS = {
    "6E": 658032,
    "1E": 109672,
    "4E": 438688,
    "2E": 219344,
}

def total_steps(model_id: str | None = None) -> int | None:
    """Training steps of ``model_id``; None for a repo we haven't catalogued."""
    model = model_id or MODEL_ID
    epochs = model.split("-")[-1]
    return MODEL_TOTAL_STEPS.get(epochs)

def checkpoint_steps(model_id: str | None = None) -> list[int] | None:
    """Every checkpoint published for ``model_id``; None if it isn't catalogued
    (then the caller has to name the steps itself, e.g. `--steps`)."""
    total = total_steps(model_id)
    if total is None:
        return None
    return list(range(0, total, CHECKPOINT_INTERVAL)) + [total]


def checkpoint_subfolder(step: int) -> str:
    return f"step{step}"


def use_model(model_id: str) -> None:
    """Repoint MODEL_ID and the checkpoint grid at ``model_id``.

    Call before the first tokenizer / checkpoint load.  Anything that captured
    CHECKPOINT_STEPS / TOTAL_STEPS as a *default argument* at import time
    (src/features/learning.py, src/features/timeline.py, src/schedule.py) keeps
    the old values, so pass those explicitly -- or use $LMENT_MODEL_ID, which
    lands before any import and needs no such care.
    """
    global MODEL_ID, CHECKPOINT_STEPS, FINAL_STEP, TOTAL_STEPS
    MODEL_ID = model_id
    CHECKPOINT_STEPS = checkpoint_steps(model_id)
    FINAL_STEP = CHECKPOINT_STEPS[-1] if CHECKPOINT_STEPS else None
    TOTAL_STEPS = FINAL_STEP


CHECKPOINT_STEPS = checkpoint_steps()
# Last published checkpoint == the last training step, so these coincide; both
# names are kept because they mean different things downstream (a subfolder to
# load vs. the horizon of the LR schedule / timeline features).
FINAL_STEP = CHECKPOINT_STEPS[-1] if CHECKPOINT_STEPS else None
TOTAL_STEPS = FINAL_STEP


# --------------------------------------------------------------------------- #
# Learning-rate schedule
# --------------------------------------------------------------------------- #
# From the LMEnt paper's training details (§B.4), which is authoritative for the
# 1B run -- NOT kas_config.json (that file is the OLMo2-170M *example*, lr=3e-4).
# §B.4: "All LMEnt models were trained ... with a global batch size of 32,768,
# rank batch size 8,192, peak learning rate of 5e-3, weight decay of 0.05, and
# 1,000 warmup steps."  OLMo-2 architecture => linear warmup + cosine decay.
# VERIFY the peak-LR superscript on the PDF (5e-3 vs 5e-4 is easy to misread);
# the LR feature only depends on the curve *shape*, so absolute scale is low-risk.
# alpha_f (cosine floor fraction) is not stated; OLMo-core CosWithWarmup defaults
# to 0.1  ->  LR_MIN = alpha_f * LR_PEAK.
LR_PEAK = 5.0e-3
LR_WARMUP_STEPS = 1000
LR_ALPHA_F = 0.1                 # VERIFY (cosine floor fraction)
LR_MIN = LR_ALPHA_F * LR_PEAK


# --------------------------------------------------------------------------- #
# PopQA cloze evaluation (olmes PopQACloze)
# --------------------------------------------------------------------------- #
# VERIFIED against dhgottesman/olmes (the fork LMEnt uses as its eval harness):
# `PopQACloze.TASK_CONFIG_DEFAULTS["num_shots"] = 15`, and the registered config
# `popqa_cloze_sitelinks` (task_name="popqa_cloze", the sitelink-stratified one
# matching the paper's popularity split) ALSO sets num_shots=15.  There is no
# zero-shot cloze config anywhere in the fork.  So 15 is the harness default for
# the *cloze* task, not just the benchmark table -- an earlier note here claimed
# otherwise and was wrong.
#
# We keep it a knob rather than a constant: n-shot reproduces LMEnt's process, 0 is
# the cleaner probe of purely parametric knowledge (no in-context answers at all).
# Override per run with `--num-shots`.
#
# 15, but for a different reason than upstream's 15.
#
# The pool holds one fact per relation (16), and a fact of relation p is shown the
# pool MINUS relation p -- so every query gets exactly 15 demonstrations and the
# relation under test is never demonstrated.  See `src.fewshot.FewShot`.
#
# That matters because a same-relation demonstration hands over the answer *type*:
# with "The color of FC Internazionale Milano is black" in context, "The color of X
# is" only has to produce some colour.  That is in-context scaffolding for exactly
# what an acquisition study wants to read out of the weights.  Dropping it keeps
# every fact symmetric (15 demos, none sharing its relation) and uses all 16 pool
# facts across the eval -- unlike upstream's `_fewshot_docs[:15]` of 16, which
# silently drops the last entry so `prop_id` 422 is evaluated but never
# demonstrated by anything.
#
# `--fewshot-keep-same-prop` restores a single fixed block (upstream's behaviour),
# in which case 16 is the natural setting.
NUM_SHOTS = 15

# Demonstrations come from a HELD-OUT pool, never from the eval set.
#
# Upstream would sample them from the data it scores: `PopQA` sets
# `fewshot_source: "popqa"` and reads the curated FEWSHOT_SOURCES list, but
# `PopQACloze` does NOT set it (while still importing FEWSHOT_SOURCES, unused --
# it looks dropped when the file was derived from popqa.py).  With
# `has_training_docs()` and `has_validation_docs()` both False, PopQACloze lands on
# `rnd.sample(self.test_docs(), k * 2)` (base_task.py:597) -- demonstrations drawn
# from the eval set with only the doc under evaluation filtered out.  Other eval
# facts still appear in context *with their gold answers*, so `correct` partly
# measures in-context copying rather than parametric recall.
#
# The fork's curated FEWSHOT_SOURCES["popqa"] is not a fix either: it is 16
# hand-pasted facts with no generating script, and measured against popqa-kas one
# has its answer inside its own question (prop 257) while two prime answers shared
# by 284 and 327 eval facts (props 182, 560).  All 16 are popqa-kas rows.  It is
# vendored in `src/fewshot_source.py` and reachable via `--fewshot-source upstream`
# for parity runs, but is not the default.
#
# So: cloze prompts + scoring from PopQACloze (upstream), demonstrations selected
# by the documented rule in `src/fewshot.py`.  The pool's subjects are dropped from
# the eval set at EVERY shot count -- 16 facts, ~0.1% -- so 0-shot and n-shot score
# an identical fact set and are directly comparable.

# Decode length / stop sequences come from olmes generation_kwargs
# (max_gen_toks=15, stop=["\n\n"]) via src.lment_popqa -- no local constant.


# --------------------------------------------------------------------------- #
# Local storage layout
# --------------------------------------------------------------------------- #
# Root for anything we download / cache, and the single place every script
# writes to.  Anchored to this file's directory rather than the cwd: the old
# "./data" default silently created a second tree when a script was launched
# from anywhere else.  Override with LMENT_DATA_ROOT.
DATA_ROOT = Path(os.environ.get(
    "LMENT_DATA_ROOT", str(Path(__file__).resolve().parent / "data"))).resolve()

# Where the LMEnt-Dataset is downloaded to (batch indices, mentions, tokenized).
DATASET_DIR = DATA_ROOT / "LMEnt-Dataset"
BATCH_INDICES_PATH = DATASET_DIR / "dataset-cache" / "batch_indices_epoch_1.npy"  # VERIFY name
MENTIONS_DIR = DATASET_DIR / "dataset-metadata"      # ReFinED + Maverick mentions
TOKENIZED_DIR = DATASET_DIR / "dataset-tokenized"    # *.csv.gz tokenized chunks
# Glob of the tokenized *.npy files that back the OLMo-core dataset used for the
# mention features (chunk text + per-chunk entity annotations).  Only needed if
# you downloaded the full corpus; override with LMENT_TOKENIZED_GLOB.  When no
# file matches (PopQA-only setup), mention features are skipped.
TOKENIZED_GLOB = os.environ.get("LMENT_TOKENIZED_GLOB", str(TOKENIZED_DIR / "*.npy"))

# Where we write derived outputs (Step A/B/C).  These used to sit under an
# extra <data>/artifacts/ level; that layer is gone and they are now siblings
# directly under DATA_ROOT, alongside corpora/ hf/ prompts/ runs/.
RECALL_DIR = DATA_ROOT / "recall"             # per-checkpoint recall / gold prob
FEATURE_DIR = DATA_ROOT / "features"          # per-fact feature tables
INTERNAL_DIR = DATA_ROOT / "internal"         # hidden states / probes / logit lens
CHUNK_INDEX_PATH = DATA_ROOT / "chunk_to_steps.pkl"  # inverted index cache

for _d in (RECALL_DIR, FEATURE_DIR, INTERNAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Eval settings
# --------------------------------------------------------------------------- #
DEVICE = os.environ.get("LMENT_DEVICE", "cuda")
DTYPE = "bfloat16"
SEED = 0

# --------------------------------------------------------------------------- #
# Internal-state extraction (scripts/eval_checkpoints.py --features)
# --------------------------------------------------------------------------- #
# Which token positions to dump the full per-layer hidden state for.  "answer"
# is the cloze's last token (predicts the object); "subject_last" feeds the
# linear probe + representation drift.  Storage per position ~= n_facts *
# (n_layers+1) * hidden * 2 bytes (f16); for LMEnt-1B (19 x 2048) that's ~76 KB
# per fact per checkpoint per position.  Add "all" handling in the script only if
# you really need every token (blows up ~10-15x).
HIDDEN_POSITIONS = ("answer", "subject_last")
# |activation| threshold for counting an MLP neuron as "active".
NEURON_EPS = 1e-3
# If set to an int k, also dump the top-k active MLP neurons per (fact, layer,
# position) -> artifacts/internal/neurons_step{N}_{pos}_idx.npy (uint16 indices,
# ranked by |activation|) and _val.npy (float16 *signed* activations).  Enables
# neuron-overlap / circuit-churn analysis; the values additionally enable
# magnitude-weighted overlap and knowledge-entropy-style measures.
# Storage ~= n_facts * n_layers * n_positions * k * (2 + 2) bytes with values,
# half that with --no-neuron-values.  For the 1452-fact x 13-checkpoint sweep at
# k=128 that is ~348 MB, ~6% of the hidden dump; k=64 -> ~174 MB.
# Choosing k: the measured participation ratio (effective neuron count, the
# mlp_participation_by_layer column) is p50 ~8, p99 ~45, p99.9 ~58 at both
# positions with intermediate_size=5632, so k=128 covers the tail with 2x
# headroom while staying clear of the ~600-1100 generically-active neurons per
# mid-stack layer.  Prefer erring high: top-32 can be derived from a top-128
# dump, but not the reverse, and re-running costs a full GPU sweep.
# None = off.  Override per-run with --top-k.
NEURON_TOPK = None
