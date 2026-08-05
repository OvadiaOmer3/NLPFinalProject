"""Dataset / model loading for the explicit-mentions evaluation.

Adapted from the main repo's src/data.py. The one real change: models can be
loaded either from a HuggingFace repo with step subfolders (the published
LMEnt checkpoints, e.g. "dhgottesman/LMEnt-170M-1E") or from a local
directory of converted checkpoints (what convert_checkpoints.py produces,
laid out the same way: <dir>/stepN/{config.json, model.safetensors, ...}).
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import torch
from datasets import load_dataset

import config
import re

# Set by eval_explicit.py before any model loads; an HF id or a local dir.
MODEL_SPEC: str = config.BASELINE_MODEL_ID


def set_model_spec(spec: str):
    global MODEL_SPEC
    MODEL_SPEC = spec
    load_tokenizer.cache_clear()


def _is_local(spec: str) -> bool:
    return os.path.isdir(spec)


def _available_steps(spec: str) -> list[int]:
    if not _is_local(spec):
        raise ValueError("step listing only supported for local checkpoint dirs")
    return sorted(int(p.name[4:]) for p in Path(spec).glob("step*") if p.is_dir())


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
# def load_popqa_df(split: str = "train"):
#     """PopQA-KAS as a DataFrame; drops the 3 rows with a null subj/obj."""
#     df = (load_dataset(config.POPQA_DATASET_ID, split=split)
#           .to_pandas().dropna(subset=["subj", "obj"]).reset_index(drop=True))
#     return df

def load_popqa(split: str = "train"):
    """Return the PopQA-KAS split as a HuggingFace Dataset."""
    return load_dataset(config.POPQA_DATASET_ID, split=split)

def get_words(x):
        remove = set(['film','movie', 'game' , 'story', 'song', 'novel', 'series', 'video', 'short' ])
        x = str(x).lower()
        x = re.sub(r'[(),]', ' ', x)  # remove parentheses and commas
        return set(x.split()) - remove
    
def load_popqa_df(split: str = "train"):
    """Same data as a pandas DataFrame (convenient for feature joins).

    Three rows have a null ``subj`` (ids 305622, 1703675, 1227912).  They render
    as "The author of nan is" and crash the subject-span lookup in
    ``subject_last_token_index``; the fact is unmeasurable either way, so drop
    them here -> 14264 rows.
    """
    df = load_popqa(split).to_pandas().dropna(subset=["subj", "obj"]).reset_index(drop=True)
    # df['possible_answers'] = df['possible_answers'].apply(tuple)
    # df = df.drop_duplicates(['question', 'possible_answers'])
    # df['possible_answers'] = df['possible_answers'].apply(list)
    # dup_mask = df.duplicated('question', keep=False)
    # df.loc[dup_mask, 'subj'] = df.loc[dup_mask, 's_wiki_title']
    # answer_in_mask = df.apply(lambda row: bool(get_words(row['subj']) & get_words(row['obj'])),axis=1)
    # df = df[~(answer_in_mask & dup_mask)]
    return df

# --------------------------------------------------------------------------- #
# Model / tokenizer
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def load_tokenizer():
    from transformers import AutoTokenizer
    if _is_local(MODEL_SPEC):
        step = _available_steps(MODEL_SPEC)[-1]
        try:
            return AutoTokenizer.from_pretrained(
                str(Path(MODEL_SPEC) / f"step{step}"), use_fast=True)
        except Exception as exc:
            # A checkpoint converted with transformers 5.x records a v5-only
            # tokenizer_class that 4.x rejects. The dolma2 tokenizer is
            # identical across every LMEnt model, so fall back to the published
            # one rather than failing the whole eval.
            print(f"[tokenizer] local load failed ({type(exc).__name__}); "
                  f"falling back to {config.TOKENIZER_ID}")
    return AutoTokenizer.from_pretrained(
        config.TOKENIZER_ID, subfolder=config.TOKENIZER_SUBFOLDER, use_fast=True)


def load_checkpoint(step: int, device: str | None = None):
    from transformers import AutoModelForCausalLM
    device = device or os.environ.get("LMENT_DEVICE", config.DEVICE)
    dtype = getattr(torch, os.environ.get("LMENT_DTYPE", config.DTYPE))
    if _is_local(MODEL_SPEC):
        src, kwargs = str(Path(MODEL_SPEC) / f"step{step}"), {}
    else:
        src, kwargs = MODEL_SPEC, {"subfolder": f"step{step}"}
    try:
        model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, **kwargs)
    model.to(device).eval()
    return model


# --------------------------------------------------------------------------- #
# Prompt construction (single-sourced from olmes via emlib.lment_popqa)
# --------------------------------------------------------------------------- #
def prompt_for(row, prefix: str = "") -> str:
    from emlib import lment_popqa
    return lment_popqa.make_prompt(row, prefix=prefix)


def zero_shot_cloze(row) -> str:
    from emlib import lment_popqa
    return lment_popqa.zero_shot_cloze(row)


def answer_token_ids(obj: str, tokenizer, prefix_space: bool = True) -> list[int]:
    text = (" " + obj) if prefix_space else obj
    return tokenizer(text, add_special_tokens=False)["input_ids"]
