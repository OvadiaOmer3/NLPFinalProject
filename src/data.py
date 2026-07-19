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
    """Load the model weights for a single checkpoint step onto `device`."""
    device = device or config.DEVICE
    dtype = getattr(torch, config.DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        subfolder=config.checkpoint_subfolder(step),
        torch_dtype=dtype,
    )
    model.to(device).eval()
    return model


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
# Cloze templates per PopQA relation.  The template ends exactly where the
# answer (object) should be generated, which is what we need for gold-answer
# probability, subject-last-token hidden states, and logit lens.
# Keyed by the `prop` string.  VERIFY the exact relation strings in the data.
RELATION_TEMPLATES: dict[str, str] = {
    "occupation":      "{subj} works as a",
    "place of birth":  "{subj} was born in the city of",
    "genre":           "The genre of {subj} is",
    "father":          "The father of {subj} is",
    "country":         "{subj} is located in the country of",
    "producer":        "The producer of {subj} is",
    "director":        "The director of {subj} is",
    "capital of":      "{subj} is the capital of",
    "screenwriter":    "The screenwriter of {subj} is",
    "composer":        "The composer of {subj} is",
    "color":           "The color of {subj} is",
    "religion":        "The religion of {subj} is",
    "sport":           "{subj} plays the sport of",
    "author":          "The author of {subj} is",
    "mother":          "The mother of {subj} is",
    "capital":         "The capital of {subj} is",
}


def cloze_prompt(subj: str, prop: str) -> str:
    """Cloze prompt that stops right before the answer span."""
    tmpl = RELATION_TEMPLATES.get(prop)
    if tmpl is None:
        # Fallback: use the natural-language question with an answer cue.
        return None
    return tmpl.format(subj=subj)


def qa_prompt(question: str) -> str:
    """Question-answer prompt used for the primary substring-match recall."""
    return f"Q: {question}\nA:"


def paraphrase_prompts(subj: str, prop: str, question: str) -> list[str]:
    """A small set of phrasings for the paraphrase-stability feature."""
    prompts = [qa_prompt(question)]
    cloze = cloze_prompt(subj, prop)
    if cloze is not None:
        prompts.append(cloze)
        prompts.append(f"{cloze} ")  # trivial surface variant
    return prompts


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
