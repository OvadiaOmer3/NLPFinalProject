"""Loading the dataset, checkpoints, tokenizer, and building prompts.

The PopQA-KAS schema (confirmed against the real parquet):

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
from pathlib import Path
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
    """Same data as a pandas DataFrame (convenient for feature joins).

    Three rows have a null ``subj`` (ids 305622, 1703675, 1227912).  They render
    as "The author of nan is" and crash the subject-span lookup in
    ``subject_last_token_index``; the fact is unmeasurable either way, so drop
    them here -> 14264 rows.
    """
    df = load_popqa(split).to_pandas().dropna(subset=["subj", "obj"]).reset_index(drop=True)
    return df

# --------------------------------------------------------------------------- #
# Model / tokenizer
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4)
def _tokenizer(model_id: str, subfolder: str):
    if config.is_local_model(model_id):
        try:
            return AutoTokenizer.from_pretrained(str(Path(model_id) / subfolder), use_fast=True)
        except Exception as exc:
            # A checkpoint converted with transformers 5.x records a v5-only
            # tokenizer_class that 4.x rejects.  The dolma2 tokenizer is identical
            # across every LMEnt model, so fall back to the published one rather
            # than failing the whole eval.
            print(f"[tokenizer] {model_id}/{subfolder} failed ({type(exc).__name__}); "
                  f"falling back to {config.TOKENIZER_ID}/{config.TOKENIZER_SUBFOLDER}")
            return _tokenizer(config.TOKENIZER_ID, config.TOKENIZER_SUBFOLDER)
    return AutoTokenizer.from_pretrained(model_id, subfolder=subfolder, use_fast=True)


def available_steps(model_id: str | None = None) -> list[int]:
    """Checkpoint steps of ``model_id`` (default config.MODEL_ID): the published
    grid for a catalogued HF release, the stepN/ dirs for a local checkpoint dir,
    [] when neither is known."""
    return config.checkpoint_steps(model_id or config.MODEL_ID) or []


def load_tokenizer(model_id: str | None = None, step: int | None = None):
    # Tokenizer is identical across checkpoints; load from the final one.  Keyed
    # on the model so a process that switches models doesn't reuse a stale one.
    # ``model_id`` may be an HF repo id or a local dir of converted checkpoints.
    model_id = model_id or config.MODEL_ID
    if step is None:
        steps = config.checkpoint_steps(model_id)
        step = steps[-1] if steps else config.FINAL_STEP
        if step is None:
            raise ValueError(f"no checkpoint grid known for {model_id!r}; pass step=")
    return _tokenizer(model_id, config.checkpoint_subfolder(step))


def _check_step(model_id: str, step: int) -> None:
    """Fail loudly on a step this repo doesn't publish.

    Without this, `from_pretrained` 404s the subfolder and transformers falls
    back to inferring the model type from the repo *name* -- "dhgottesman"
    contains "esm" -- so a wrong step surfaces as an unrelated
    "EsmConfig ... for this kind of AutoModel: AutoModelForCausalLM".
    """
    steps = config.checkpoint_steps(model_id)
    if steps is not None and step not in steps:
        if config.is_local_model(model_id):
            raise ValueError(f"{model_id} has no step{step}/ directory; found {steps}")
        raise ValueError(
            f"{model_id} has no step{step} checkpoint: it publishes step0 .. "
            f"step{steps[-2]} every {config.CHECKPOINT_INTERVAL} plus the final "
            f"step{steps[-1]}. (Different LMEnt releases have different finals -- "
            f"109672 for the 1E models, 658032 for 1B-6E.)"
        )


def load_checkpoint(step: int, device: str | None = None, model_id: str | None = None):
    """Load the model weights for a single checkpoint step onto `device`.

    ``model_id`` is an HF repo with ``stepN`` subfolders (the published LMEnt
    checkpoints) or a local directory laid out the same way
    (``<dir>/stepN/{config.json, model.safetensors, ...}``, what
    explicit_mentions/convert_checkpoints.py produces).

    Honors LMENT_DEVICE / LMENT_DTYPE env overrides (e.g. cpu / float32 for a
    machine without CUDA).  Uses the `dtype=` kwarg (transformers >= 5) with a
    `torch_dtype=` fallback for older versions.
    """
    import os
    model_id = model_id or config.MODEL_ID
    _check_step(model_id, step)
    device = device or os.environ.get("LMENT_DEVICE", config.DEVICE)
    dtype = getattr(torch, os.environ.get("LMENT_DTYPE", config.DTYPE))
    sub = config.checkpoint_subfolder(step)
    if config.is_local_model(model_id):
        src, kw = str(Path(model_id) / sub), {}
    else:
        src, kw = model_id, {"subfolder": sub}
    try:
        model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, **kw)
    model.to(device).eval()
    return model


# --------------------------------------------------------------------------- #
# Prompt construction  (single-sourced from olmes via src.lment_popqa)
# --------------------------------------------------------------------------- #
# Prompts follow the paper's methodology (cloze, few-shot) and are built by the
# olmes PopQACloze task.  These thin wrappers keep call sites readable.
def prompt_for(row, prefix: str = "") -> str:
    """Primary recall prompt: demonstration block + this fact's cloze.

    ``prefix`` comes from ``src.fewshot.FewShot.prefix``; the default ""  gives
    the bare zero-shot cloze.
    """
    from src import lment_popqa
    return lment_popqa.make_prompt(row, prefix=prefix)


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

    ``rfind``, not ``find``: the fact's own cloze is always the *tail* of the
    prompt, and under a few-shot prefix the subject string can also occur inside
    the demonstration block (measured on the 15-shot top-decile run: 13 of 1452
    facts -- ``Milan``, ``Delhi``, ``Papua New Guinea``, and short strings like
    ``M`` / ``Ran``).  ``find`` would silently return a *demo* token there, so
    the hidden state and neuron stats would describe the wrong entity.  On a
    zero-shot cloze the subject occurs once and the two agree.
    """
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    char_start = prompt.rfind(subj)
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
