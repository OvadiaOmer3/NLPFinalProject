"""Per-checkpoint recall: substring-match correctness + gold-answer probability.

For each (fact, checkpoint) we record:
  * ``correct``   -- olmes PopQACloze exact-match on a greedy generation from the
                     cloze prompt.  Zero-shot by default (``config.NUM_SHOTS=0``),
                     matching the paper's knowledge eval; scoring + prompt +
                     decode length all come from ``src.lment_popqa``, not from
                     locally guessed constants.
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

from src import data as D
from src import lment_popqa


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
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, pad_token_id=tokenizer.eos_token_id,
                         **lment_popqa.gen_kwargs())
    gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return lment_popqa.truncate_at_stop(gen)


@torch.no_grad()
def eval_checkpoint(model, tokenizer, df) -> "pd.DataFrame":
    """Evaluate every fact at a single loaded checkpoint.

    Returns a DataFrame with columns:
    id, correct, excluded, gold_prob, prompt, generation.
    """
    import pandas as pd
    rows = []
    for _, r in df.iterrows():
        # Primary recall: cloze prompt (zero-shot by default), olmes scoring.
        prompt = D.prompt_for(r)
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
