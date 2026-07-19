"""Per-checkpoint recall: substring-match correctness + gold-answer probability.

For each (fact, checkpoint) we record two signals:
  * ``correct``  -- greedy decode contains any acceptable answer (PopQA style)
  * ``gold_prob`` -- probability mass the model puts on the gold object tokens
                     right after the prompt (the finer-grained signal from the
                     proposal footnote).

The output of ``eval_checkpoint`` over all checkpoints is the acquisition
timeline that every learning-dynamics feature is derived from.
"""
from __future__ import annotations

import numpy as np
import torch

import config
from src import data as D


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def acceptable_answers(row) -> list[str]:
    ans = [row["obj"]]
    poss = row.get("possible_answers")
    if poss is not None:
        ans.extend(list(poss))
    o_alias = row.get("o_aliases")
    if o_alias is not None:
        ans.extend(list(o_alias))
    return [_normalize(a) for a in ans if isinstance(a, str) and a]


@torch.no_grad()
def gold_answer_logprob(model, tokenizer, prompt: str, obj: str) -> float:
    """Sum of per-token log-probs of the gold answer tokens after the prompt,
    returned as a single sequence probability (exp of the mean log-prob so the
    value is comparable across answers of different length)."""
    device = model.device
    ans_ids = D.answer_token_ids(obj, tokenizer)
    if not ans_ids:
        return float("nan")
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    ans_t = torch.tensor([ans_ids], device=device)
    full = torch.cat([prompt_ids, ans_t], dim=1)
    logits = model(full).logits          # [1, T, V]
    # positions predicting the answer tokens are prompt_len-1 .. end-1
    start = prompt_ids.shape[1] - 1
    logprobs = torch.log_softmax(logits[0, start:start + len(ans_ids)], dim=-1)
    tok_lp = logprobs[torch.arange(len(ans_ids)), torch.tensor(ans_ids, device=device)]
    mean_lp = tok_lp.mean().item()       # length-normalized
    return float(np.exp(mean_lp))


@torch.no_grad()
def greedy_correct(model, tokenizer, prompt: str, answers: list[str]) -> bool:
    device = model.device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=config.MAX_NEW_TOKENS,
                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    gen = _normalize(gen)
    return any(a in gen for a in answers)


@torch.no_grad()
def eval_checkpoint(model, tokenizer, df) -> "pd.DataFrame":
    """Evaluate every fact at a single loaded checkpoint.

    Returns a DataFrame with columns: id, correct, gold_prob.
    """
    import pandas as pd
    rows = []
    for _, r in df.iterrows():
        # Primary recall uses the QA prompt; gold_prob uses the cloze template
        # when available (cleaner answer position), else the QA prompt.
        qa = D.qa_prompt(r["question"])
        cloze = D.cloze_prompt(r["subj"], r["prop"]) or qa
        answers = acceptable_answers(r)
        rows.append({
            "id": r["id"],
            "correct": greedy_correct(model, tokenizer, qa, answers),
            "gold_prob": gold_answer_logprob(model, tokenizer, cloze, r["obj"]),
        })
    return pd.DataFrame(rows)
