"""Central configuration for the explicit-mentions experiment.

Everything in this directory is self-contained: it reads the LMEnt corpus
straight from the authors' lab directory (read-only) and writes a rewritten
corpus + trained models + eval outputs under EXPLICIT_ROOT (the shared
NLPFinalProject/data tree).

Override any path with the environment variables noted below.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Source corpus (read-only, the authors' copy on the lab filesystem)
# --------------------------------------------------------------------------- #
# dataset-tokenized/part-N-00000.npy      raw little-endian uint32 dolma2 tokens
#                                         (no numpy header), docs concatenated,
#                                         each doc terminated by EOS (100257)
# dataset-cache/dataset-metadata/part-N-00000.csv
#                                         one row per document:
#                                         start,end,id,src,loc,title,entities,offsets
SRC_ROOT = Path(os.environ.get(
    "LMENT_SRC_ROOT", "/home/morg/students/gottesman3/LMEnt"))
SRC_TOKENIZED_DIR = SRC_ROOT / "dataset-tokenized"
SRC_METADATA_DIR = SRC_ROOT / "dataset-cache" / "dataset-metadata"

# The authors' OLMo-core fork (training code + HF conversion). We import/run it
# in place; nothing is written into it (save_folder / work_dir point at ours).
OLMO_CORE_PATH = Path(os.environ.get(
    "OLMO_CORE_PATH", "/home/morg/students/gottesman3/LMEnt/OLMo-core"))

PARTS = list(range(8))                     # part-0-00000 ... part-7-00000
PART_TEMPLATE = "part-{part}-00000"

# --------------------------------------------------------------------------- #
# Output layout (everything we produce)
# --------------------------------------------------------------------------- #
# The shared NLPFinalProject/data tree -- the same root the main pipeline's
# config.DATA_ROOT points at, so corpora/ hf/ runs/ and the eval outputs all
# live in one place instead of a separate explicit_mentions_data/ tree.
# Anchored to this file's location (../data) so it survives the package being
# moved; override with $EXPLICIT_ROOT to point at a copy elsewhere.
EXPLICIT_ROOT = Path(os.environ.get(
    "EXPLICIT_ROOT",
    str(Path(__file__).resolve().parent.parent / "data"))).resolve()

# The rewritten corpus, mirroring the source layout so OLMo-core's
# NumpyKASVSLDataset consumes it unchanged:
#   <corpus>/dataset-tokenized/part-N-00000.npy         (+ .csv.gz for bucketing)
#   <corpus>/dataset-cache/dataset-metadata/part-N-00000.csv
# Each rewrite configuration gets its own subdirectory ("corpus tag").
CORPORA_DIR = EXPLICIT_ROOT / "corpora"
RUNS_DIR = EXPLICIT_ROOT / "runs"          # OLMo-core save_folders
HF_DIR = EXPLICIT_ROOT / "hf"              # converted HF checkpoints
# Eval outputs.  There is no longer an artifacts/ level: everything the eval
# scripts produce goes under recall/, matching the main pipeline's
# config.RECALL_DIR so both trees write to the same directory.
RECALL_DIR = EXPLICIT_ROOT / "recall"
PROMPTS_DIR = EXPLICIT_ROOT / "prompts"    # few-shot prompt parquets


def corpus_dir(tag: str) -> Path:
    return CORPORA_DIR / tag


def corpus_tokenized_dir(tag: str) -> Path:
    return corpus_dir(tag) / "dataset-tokenized"


def corpus_metadata_dir(tag: str) -> Path:
    return corpus_dir(tag) / "dataset-cache" / "dataset-metadata"


def corpus_work_dir(tag: str) -> Path:
    # OLMo-core work_dir: it expects dataset-metadata/ inside and creates
    # dataset-common/ etc. next to it.
    return corpus_dir(tag) / "dataset-cache"


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
# The corpus uses allenai/dolma2-tokenizer. The LMEnt model repos ship the
# identical tokenizer, and dhgottesman/LMEnt-1B-6E is already in the local HF
# cache, so we default to loading it from there (works offline on compute
# nodes). Override with EXPLICIT_TOKENIZER (an HF id or local path).
TOKENIZER_ID = os.environ.get("EXPLICIT_TOKENIZER", "dhgottesman/LMEnt-1B-6E")
TOKENIZER_SUBFOLDER = os.environ.get("EXPLICIT_TOKENIZER_SUBFOLDER", "step658032")
EOS_TOKEN_ID = 100257                      # dolma2 <|endoftext|>
TOKEN_DTYPE = "<u4"                        # little-endian uint32, headerless

# --------------------------------------------------------------------------- #
# Mention-rewriting defaults (see build_explicit_corpus.py --help)
# --------------------------------------------------------------------------- #
# A mention is IMPLICIT when its best candidate has no hyperlink and no
# entity-linking evidence (coref / coref_cluster only). We only rewrite
# implicit mentions whose coref evidence clears this threshold -- 0.6 matches
# the retrieval threshold used in the paper (S5.2: C >= 0.6).
MIN_COREF_SCORE = 0.6
# Candidates below this aggregated score are never trusted for rewriting.
MIN_AGGREGATED_SCORE = 0.0

# --------------------------------------------------------------------------- #
# PopQA (for --scope popqa and for evaluation)
# --------------------------------------------------------------------------- #
POPQA_DATASET_ID = "dhgottesman/popqa-kas"

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
# Baseline published models, for comparing against our retrained ones.
BASELINE_MODEL_ID = "dhgottesman/LMEnt-170M-1E"
DEVICE = os.environ.get("LMENT_DEVICE", "cuda")
DTYPE = os.environ.get("LMENT_DTYPE", "bfloat16")
SEED = 0
NUM_SHOTS = 5
