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
MODEL_ID = "dhgottesman/LMEnt-1B-6E"          # 1B model, single epoch
POPQA_DATASET_ID = "dhgottesman/popqa-kas"    # entity-annotated PopQA
LMENT_DATASET_ID = "dhgottesman/LMEnt-Dataset"  # corpus + batch indices + mentions

# --------------------------------------------------------------------------- #
# Checkpoints  (subfolders inside MODEL_ID)
# --------------------------------------------------------------------------- #
# From the model repo tree: step0, step10000 ... step100000, step109672 (final).
CHECKPOINT_STEPS = [
    0, 10000, 20000, 30000, 40000, 50000,
    60000, 70000, 80000, 90000, 100000,
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
# The paper evaluates knowledge with ZERO-SHOT cloze prompts (§5.1, §B.6): models
# are not instruction-tuned, so each question becomes a cloze whose answer is the
# next phrase; the "learned" criterion is just completing that cloze correctly.
# So NUM_SHOTS = 0 matches the paper's knowledge-acquisition / learning-dynamics
# analysis (which is what this project studies).  The 15-shot setting is the
# OLMES *benchmark* default (for leaderboard comparison in §5.1's baseline table)
# -- switch to it only to reproduce those benchmark numbers.  The seed is held
# FIXED across checkpoints so any demonstrations are identical at every step.
NUM_SHOTS = 0
FEWSHOT_SEED = 1234
# Decode length / stop sequences come from olmes generation_kwargs
# (max_gen_toks=15, stop=["\n\n"]) via src.lment_popqa -- no local constant.


# --------------------------------------------------------------------------- #
# Local storage layout
# --------------------------------------------------------------------------- #
# Root for anything we download / cache.  Override with LMENT_DATA_ROOT.
DATA_ROOT = Path(os.environ.get("LMENT_DATA_ROOT", "./data")).resolve()

# Where the LMEnt-Dataset is downloaded to (batch indices, mentions, tokenized).
DATASET_DIR = DATA_ROOT / "LMEnt-Dataset"
BATCH_INDICES_PATH = DATASET_DIR / "dataset-cache" / "batch_indices_epoch_1.npy"
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
