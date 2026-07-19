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
MODEL_ID = "dhgottesman/LMEnt-1B-1E"          # 1B model, single epoch
POPQA_DATASET_ID = "dhgottesman/popqa-kas"    # entity-annotated PopQA
LMENT_DATASET_ID = "dhgottesman/LMEnt-Dataset"  # corpus + batch indices + mentions

# --------------------------------------------------------------------------- #
# Checkpoints  (subfolders inside MODEL_ID)
# --------------------------------------------------------------------------- #
# From the model repo tree: step0, step10000 ... step100000, step109672 (final).
CHECKPOINT_STEPS = [
    0, 10000, 20000, 30000, 40000, 50000,
    60000, 70000, 80000, 90000, 100000, 109672,
]
FINAL_STEP = CHECKPOINT_STEPS[-1]              # total training steps (1 epoch)
TOTAL_STEPS = FINAL_STEP


def checkpoint_subfolder(step: int) -> str:
    return f"step{step}"


# --------------------------------------------------------------------------- #
# Learning-rate schedule
# --------------------------------------------------------------------------- #
# OLMo-core style linear-warmup + cosine decay.  The exact peak LR / warmup are
# read from the training config if available; the constants below are the
# published OLMo-1B defaults and MUST be confirmed against LMEnt's config.
# VERIFY: peak_lr, warmup_steps, min_lr against the checkpoint's config / paper.
LR_PEAK = 4.0e-4          # VERIFY
LR_WARMUP_STEPS = 2000    # VERIFY
LR_MIN = 0.0              # VERIFY (cosine floor)


# --------------------------------------------------------------------------- #
# Local storage layout
# --------------------------------------------------------------------------- #
# Root for anything we download / cache.  Override with LMENT_DATA_ROOT.
DATA_ROOT = Path(os.environ.get("LMENT_DATA_ROOT", "./data")).resolve()

# Where the LMEnt-Dataset is downloaded to (batch indices, mentions, tokenized).
DATASET_DIR = DATA_ROOT / "LMEnt-Dataset"
BATCH_INDICES_PATH = DATASET_DIR / "dataset-cache" / "batch_indices.npy"  # VERIFY name
MENTIONS_DIR = DATASET_DIR / "dataset-metadata"      # ReFinED + Maverick mentions
TOKENIZED_DIR = DATASET_DIR / "dataset-tokenized"    # *.csv.gz tokenized chunks

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
MAX_NEW_TOKENS = 16          # greedy decode length for substring-match recall
SEED = 0
