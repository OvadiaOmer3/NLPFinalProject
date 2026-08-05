"""Per-checkpoint recall: substring-match correctness + gold-answer probability.

For each (fact, checkpoint) we record:
  * ``correct``   -- olmes PopQACloze exact-match on a greedy generation from the
                     cloze prompt.  15-shot by default (``config.NUM_SHOTS``, the
                     upstream PopQACloze default), with demonstrations from the
                     fork's curated ``FEWSHOT_SOURCES["popqa"]``; pass 0 for the
                     bare cloze.  Scoring + prompt + decode length all come from
                     ``src.lment_popqa``, not from locally guessed constants.
  * ``excluded``  -- olmes drops facts whose answer already appears in the query;
                     we flag them so EDA can filter them out.
  * ``gold_prob`` -- probability mass on the gold object tokens right after the
                     *zero-shot* cloze (the finer-grained signal from the
                     proposal footnote).

The sweep over all checkpoints is the acquisition timeline every learning-
dynamics feature is derived from.
"""
from __future__ import annotations

import numpy as np
import torch

from emlib import data as D
from emlib import lment_popqa


@torch.no_grad()
def gold_answer_logprob(model, tokenizer, prompt: str, obj: str) -> float:
    """Length-normalized probability of the gold answer tokens after `prompt`
    (exp of the mean per-token log-prob, comparable across answer lengths)."""
    device = model.device
    ans_ids = D.answer_token_ids(obj, tokenizer)
    if not ans_ids:
        return float("nan")
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    ans_t = torch.tensor([ans_ids], device=device)
    full = torch.cat([prompt_ids, ans_t], dim=1)
    logits = model(full).logits          # [1, T, V]
    start = prompt_ids.shape[1] - 1      # positions predicting the answer tokens
    logprobs = torch.log_softmax(logits[0, start:start + len(ans_ids)], dim=-1)
    tok_lp = logprobs[torch.arange(len(ans_ids)), torch.tensor(ans_ids, device=device)]
    return float(np.exp(tok_lp.mean().item()))


@torch.no_grad()
def greedy_generation(model, tokenizer, prompt: str) -> str:
    """Greedy decode with olmes generation settings; truncated at the stop seq."""
    device = model.device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    # Some tokenizer configs (e.g. a checkpoint converted by transformers 5.x)
    # also return token_type_ids, which generate() refuses. Pass only what the
    # model consumes.
    gen_inputs = {k: v for k, v in enc.items()
                  if k in ("input_ids", "attention_mask")}
    out = model.generate(**gen_inputs, pad_token_id=tokenizer.eos_token_id,
                         **lment_popqa.gen_kwargs())
    gen = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return lment_popqa.truncate_at_stop(gen)


@torch.no_grad()
def eval_checkpoint(model, tokenizer, df, fewshot=None) -> "pd.DataFrame":
    """Evaluate every fact at a single loaded checkpoint.

    ``fewshot`` is a ``src.fewshot.FewShot`` (or None for the bare cloze).  Note
    ``gold_prob`` is always measured on the *zero-shot* cloze regardless, so it
    stays comparable across shot counts (see below).

    Returns a DataFrame with columns:
    id, correct, excluded, gold_prob, prompt, generation.
    """
    import pandas as pd
    rows = []
    for _, r in df.iterrows():
        # Primary recall: cloze prompt (few-shot block prepended), olmes scoring.
        prompt = D.prompt_for(r, prefix=fewshot.prefix_for(r) if fewshot else "")
        gen = greedy_generation(model, tokenizer, prompt)
        # gold_prob: zero-shot cloze so the answer position is unconditioned.
        zcloze = D.zero_shot_cloze(r)
        rows.append({
            "id": r["id"],
            "correct": lment_popqa.is_correct(r, gen),
            "excluded": lment_popqa.is_excluded(r),
            "gold_prob": gold_answer_logprob(model, tokenizer, zcloze, r["obj"]),
            "prompt": prompt,          # exact text fed to the model for `correct`
            "generation": gen,         # model's decoded answer (post stop-truncation)
        })
    return pd.DataFrame(rows)
