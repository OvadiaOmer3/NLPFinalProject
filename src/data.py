"""Loading the dataset, checkpoints, tokenizer, and building prompts.

The PopQA-KAS schema (VERIFY against the real parquet ``.features``):

    id, subj, prop, obj, subj_id, prop_id, obj_id,
    s_aliases, o_aliases, s_uri, o_uri, s_wiki_title, o_wiki_title,
    s_pop, o_pop, question, possible_answers,
    subject_chunks, subject_num_chunks,
    answer_chunks,  answer_num_chunks,
    shared_chunks,  num_shared_chunks

`*_chunks` are list[int] of chunk IDs; `shared_chunks` are the chunks that
contain BOTH subject and answer -- i.e. the places the fact itself could be
learned.
"""
from __future__ import annotations

import functools
from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_popqa(split: str = "train"):
    """Return the PopQA-KAS split as a HuggingFace Dataset."""
    return load_dataset(config.POPQA_DATASET_ID, split=split)


def load_popqa_df(split: str = "train"):
    """Same data as a pandas DataFrame (convenient for feature joins)."""
    return load_popqa(split).to_pandas()


# --------------------------------------------------------------------------- #
# Model / tokenizer
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def load_tokenizer():
    # Tokenizer is identical across checkpoints; load from the final one.
    return AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        subfolder=config.checkpoint_subfolder(config.FINAL_STEP),
        use_fast=True,
    )


def load_checkpoint(step: int, device: str | None = None):
    """Load the model weights for a single checkpoint step onto `device`.

    Honors LMENT_DEVICE / LMENT_DTYPE env overrides (e.g. cpu / float32 for a
    machine without CUDA).  Uses the `dtype=` kwarg (transformers >= 5) with a
    `torch_dtype=` fallback for older versions.
    """
    import os
    device = device or os.environ.get("LMENT_DEVICE", config.DEVICE)
    dtype = getattr(torch, os.environ.get("LMENT_DTYPE", config.DTYPE))
    sub = config.checkpoint_subfolder(step)
    try:
        model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, subfolder=sub, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, subfolder=sub, torch_dtype=dtype)
    model.to(device).eval()
    return model


# --------------------------------------------------------------------------- #
# Prompt construction  (single-sourced from olmes via src.lment_popqa)
# --------------------------------------------------------------------------- #
# Prompts follow the paper's methodology (cloze, few-shot) and are built by the
# olmes PopQACloze task.  These thin wrappers keep call sites readable.
def prompt_for(row) -> str:
    """Primary recall prompt: the few-shot cloze block (olmes)."""
    from src import lment_popqa
    return lment_popqa.make_prompt(row)


def zero_shot_cloze(row) -> str:
    """Unconditioned cloze that stops exactly before the answer.  Used for
    gold_prob, logit lens, and hidden-state extraction."""
    from src import lment_popqa
    return lment_popqa.zero_shot_cloze(row)


def subject_last_token_index(prompt: str, subj: str, tokenizer) -> int | None:
    """Index (in the tokenized prompt) of the subject's final token.

    Used for hidden-state probing / logit lens / drift.  We locate the subject
    string inside the prompt and map the character span to token positions via
    the fast tokenizer's offset mapping.
    """
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    char_start = prompt.find(subj)
    if char_start < 0:
        return None
    char_end = char_start + len(subj)
    last_idx = None
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a == b:            # special tokens have empty spans
            continue
        if a < char_end and b > char_start:   # token overlaps the subject span
            last_idx = i
    return last_idx


def answer_token_ids(obj: str, tokenizer, prefix_space: bool = True) -> list[int]:
    """Token ids for the gold answer as it would appear after the prompt."""
    text = (" " + obj) if prefix_space else obj
    return tokenizer(text, add_special_tokens=False)["input_ids"]
