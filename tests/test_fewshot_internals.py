"""Tests for measuring the internals on a few-shot prompt (--internal-fewshot).

``extract_fact`` used to hardcode ``D.zero_shot_cloze(row)``, so a 15-shot run
scored ``correct`` on one prompt and every internal feature on another.  It now
takes an explicit ``prompt=``.  Two things have to hold:

  * the DEFAULT is still the zero-shot cloze -- every artifact produced so far
    depends on the answer position being unconditioned, and a silent change here
    would invalidate them without changing a filename;
  * under a few-shot prefix the subject is located in the fact's OWN cloze.
    ``subject_last_token_index`` used ``find``, which returns the FIRST match;
    on the real 15-shot top-decile run 13 of 1452 subjects (``Milan``, ``Delhi``,
    ``Papua New Guinea``, and short strings like ``M`` / ``Ran``) also occur in
    the demonstration block, so ``find`` pointed the hidden state and the neuron
    stats at a demo token.  It uses ``rfind`` now.

The model here is a fake: these are prompt/indexing contracts, not numerics, so
the test needs no checkpoint and no network.

Run:  python NLPFinalProject/tests/test_fewshot_internals.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src.features import internal

VOCAB, HIDDEN, INTER, NLAYERS = 64, 8, 16, 3
PREFIX = ("The capital of Milan is Rome\n\n"
          "The occupation of Guinea Smith is singer\n\n")
CLOZE = "The occupation of Milan is"
ROW = {"subj": "Milan", "obj": "actor", "question": "What is Milan's occupation?"}


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeEncoding(dict):
    """Just enough of a fast-tokenizer BatchEncoding: ids + char offsets."""

    @property
    def input_ids(self):
        return torch.tensor([self["input_ids"]])


class FakeTokenizer:
    """Whitespace tokenizer with real character offsets.

    One token per whitespace-delimited word, plus a BOS with an empty span, so
    ``subject_last_token_index``'s offset walk is exercised exactly as the fast
    tokenizer drives it.
    """

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False,
                 add_special_tokens=True):
        ids, offs, i = [], [], 0
        if add_special_tokens:
            ids.append(0)
            offs.append((0, 0))                    # special tokens have empty spans
        for word in text.split(" "):
            start = text.index(word, i)
            i = start + len(word)
            ids.append(1 + (hash(word) % (VOCAB - 1)))
            offs.append((start, i))
        enc = FakeEncoding(input_ids=ids)
        if return_offsets_mapping:
            enc["offset_mapping"] = offs
        return enc


class FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(INTER, HIDDEN)


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeMLP()


class FakeInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(FakeLayer() for _ in range(NLAYERS))
        self.norm = nn.Identity()


class FakeOut:
    def __init__(self, hidden_states, logits):
        self.hidden_states = hidden_states
        self.logits = logits


class FakeModel(nn.Module):
    """Records the token count it was called with; drives the mlp hooks."""

    def __init__(self):
        super().__init__()
        self.model = FakeInner()
        self.lm_head = nn.Linear(HIDDEN, VOCAB)
        self.seen_len = None

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, ids, output_hidden_states=False):
        t = ids.shape[1]
        self.seen_len = t
        g = torch.Generator().manual_seed(t)
        for lyr in self.model.layers:              # fire the pre-hooks extract_fact set
            lyr.mlp.down_proj(torch.randn(1, t, INTER, generator=g))
        hs = tuple(torch.randn(1, t, HIDDEN, generator=g) for _ in range(NLAYERS + 1))
        return FakeOut(hs, torch.randn(1, t, VOCAB, generator=g))


def _fake_answer_ids(obj, tokenizer, prefix_space=True):
    return [7, 8]


# --------------------------------------------------------------------------- #
# subject_last_token_index
# --------------------------------------------------------------------------- #
def test_subject_resolves_inside_the_query_not_the_demos():
    tok = FakeTokenizer()
    prompt = PREFIX + CLOZE
    idx = D.subject_last_token_index(prompt, "Milan", tok)
    offs = tok(prompt, return_offsets_mapping=True)["offset_mapping"]
    assert offs[idx][0] >= len(PREFIX), (
        f"subject landed at char {offs[idx][0]}, inside the demo block "
        f"(query starts at {len(PREFIX)})")
    assert prompt[offs[idx][0]:offs[idx][1]] == "Milan"
    print("ok  subject resolves in the query, not the demo block")


def test_substring_hit_in_demos_is_not_taken():
    # "Guinea Smith" in the prefix contains the query subject as a substring --
    # the `Papua New Guinea` / `Guinea` collision from the real run.
    tok = FakeTokenizer()
    prompt = PREFIX + "The occupation of Guinea is"
    idx = D.subject_last_token_index(prompt, "Guinea", tok)
    offs = tok(prompt, return_offsets_mapping=True)["offset_mapping"]
    assert offs[idx][0] >= len(PREFIX), "took the demo's `Guinea Smith` occurrence"
    print("ok  a substring hit inside a demo is not taken")


def test_zero_shot_unchanged_and_missing_subject_is_none():
    tok = FakeTokenizer()
    idx = D.subject_last_token_index(CLOZE, "Milan", tok)
    offs = tok(CLOZE, return_offsets_mapping=True)["offset_mapping"]
    assert CLOZE[offs[idx][0]:offs[idx][1]] == "Milan"
    assert D.subject_last_token_index(CLOZE, "Nowhere", tok) is None
    print("ok  zero-shot path unchanged; absent subject -> None")


# --------------------------------------------------------------------------- #
# extract_fact prompt threading
# --------------------------------------------------------------------------- #
def _run(prompt, monkeypatched):
    model, tok = FakeModel(), FakeTokenizer()
    feats, hidden, _ = internal.extract_fact(
        model, tok, ROW, positions=("answer", "subject_last"), prompt=prompt)
    return model, feats, hidden


def test_default_prompt_is_still_the_zero_shot_cloze():
    """The contract every existing artifact rests on."""
    calls = []
    real_cloze, real_ids = D.zero_shot_cloze, D.answer_token_ids
    D.zero_shot_cloze = lambda row: (calls.append(row) or CLOZE)
    D.answer_token_ids = _fake_answer_ids
    try:
        model, feats, hidden = _run(None, True)
        assert len(calls) == 1, "extract_fact stopped falling back to the zero-shot cloze"
        # BOS + 5 cloze words + 2 gold answer tokens
        assert model.seen_len == 1 + len(CLOZE.split(" ")) + 2, model.seen_len
        assert feats["subject_found"] is True
        assert hidden["answer"].shape == (NLAYERS + 1, HIDDEN)
    finally:
        D.zero_shot_cloze, D.answer_token_ids = real_cloze, real_ids
    print("ok  prompt=None still means the zero-shot cloze")


def test_explicit_prompt_is_used_verbatim():
    calls = []
    real_cloze, real_ids = D.zero_shot_cloze, D.answer_token_ids
    D.zero_shot_cloze = lambda row: (calls.append(row) or CLOZE)
    D.answer_token_ids = _fake_answer_ids
    try:
        few = PREFIX + CLOZE
        model, feats, hidden = _run(few, True)
        assert not calls, "the zero-shot cloze was built even though a prompt was given"
        assert model.seen_len == 1 + len(few.split(" ")) + 2, model.seen_len
        assert feats["subject_found"] is True
        assert hidden["subject_last"].shape == (NLAYERS + 1, HIDDEN)
        assert np.isfinite(hidden["subject_last"]).all()
    finally:
        D.zero_shot_cloze, D.answer_token_ids = real_cloze, real_ids
    print("ok  an explicit prompt is used verbatim, and both positions resolve")


if __name__ == "__main__":
    test_subject_resolves_inside_the_query_not_the_demos()
    test_substring_hit_in_demos_is_not_taken()
    test_zero_shot_unchanged_and_missing_subject_is_none()
    test_default_prompt_is_still_the_zero_shot_cloze()
    test_explicit_prompt_is_used_verbatim()
    print("\nall few-shot-internals tests passed")
