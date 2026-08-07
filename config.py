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
# Override with LMENT_MODEL_ID to evaluate a different LMEnt release (e.g.
# dhgottesman/LMEnt-170M-6E).  All the 6E repos publish the same checkpoint grid
# -- step0, step10000 ... step650000, step658032 -- so CHECKPOINT_STEPS and the
# --steps grids in the launch scripts stay valid across model sizes.
MODEL_ID = os.environ.get("LMENT_MODEL_ID", "dhgottesman/LMEnt-1B-6E")  # 6-epoch 1B
POPQA_DATASET_ID = "dhgottesman/popqa-kas"    # entity-annotated PopQA
LMENT_DATASET_ID = "dhgottesman/LMEnt-Dataset"  # corpus + batch indices + mentions

# --------------------------------------------------------------------------- #
# Checkpoints  (subfolders inside MODEL_ID)
# --------------------------------------------------------------------------- #
# From the model repo tree: step0, step10000 ... step100000, step109672 (final).
CHECKPOINT_STEPS = [
    0, 10000, 20000, 30000, 40000, 50000,
    60000, 70000, 80000, 90000, 100000
]
FINAL_STEP = CHECKPOINT_STEPS[-1]              # total training steps (1 epoch)
TOTAL_STEPS = 658032


def checkpoint_subfolder(step: int) -> str:
    return f"step{step}"


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
# Root for anything we download / cache.  Override with LMENT_DATA_ROOT.
DATA_ROOT = Path(os.environ.get("LMENT_DATA_ROOT", "./data")).resolve()

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

# Where we write derived artifacts (Step A/B/C outputs).
ARTIFACT_DIR = DATA_ROOT / "artifacts"
RECALL_DIR = ARTIFACT_DIR / "recall"          # per-checkpoint recall / gold prob
FEATURE_DIR = ARTIFACT_DIR / "features"       # per-fact feature tables
INTERNAL_DIR = ARTIFACT_DIR / "internal"      # hidden states / probes / logit lens
CHUNK_INDEX_PATH = ARTIFACT_DIR / "chunk_to_steps.pkl"  # inverted index cache

for _d in (ARTIFACT_DIR, RECALL_DIR, FEATURE_DIR, INTERNAL_DIR):
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
# (n_layers+1) * hidden * 2 bytes (f16); for LMEnt-1B (17 x 2048) that's ~68 KB
# per fact per checkpoint per position.  Add "all" handling in the script only if
# you really need every token (blows up ~10-15x).
HIDDEN_POSITIONS = ("answer", "subject_last")
# |activation| threshold for counting an MLP neuron as "active".
NEURON_EPS = 1e-3
# If set to an int k, also dump the indices of the top-k active MLP neurons per
# (fact, layer, position) as uint16 -> artifacts/internal/neurons_step{N}_*.npy.
# Enables neuron-overlap / circuit-churn analysis.  Storage ~= n_facts * n_layers
# * n_positions * k * 2 bytes (~4 KB per fact per checkpoint at k=64, both
# positions -- ~3% of the hidden dump).  None = off.  Override per-run with --top-k.
NEURON_TOPK = None
