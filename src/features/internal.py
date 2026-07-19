"""Group 5 - "Inside the model: where and how the fact is stored".

These features require running the checkpoint yourself (GPU).  They ask *how*
the fact is represented, not just *whether* the answer is correct:

  * hidden-state probe      -- is the answer linearly readable from the subject
                               token?  (ties to Merullo et al.)
  * logit-lens depth        -- first layer where the gold answer becomes top-1
  * answer margin           -- gold logit minus runner-up logit
  * representation drift     -- cosine movement of the subject token between
                               consecutive checkpoints
  * paraphrase stability    -- does the fact survive rephrasing?

All functions take an already-loaded model.  Hidden states are extracted at the
subject's last token (for probe/drift) or the last prompt token (for logit lens
/ margin, i.e. the position that predicts the answer).
"""
from __future__ import annotations

import numpy as np
import torch

import config
from src import data as D
from src import recall


# --------------------------------------------------------------------------- #
# Model internals accessors (OLMo-core architecture; adjust if VERIFY differs)
# --------------------------------------------------------------------------- #
def _final_norm_and_head(model):
    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None) or getattr(base, "final_layer_norm", None)
    head = getattr(model, "lm_head", None)
    if head is None:                      # tied embeddings fallback
        head = model.get_output_embeddings()
    return norm, head


@torch.no_grad()
def _hidden_states(model, tokenizer, prompt):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model(ids, output_hidden_states=True)
    # tuple: (embeddings, layer1, ..., layerL); each [1, T, H]
    return ids, out.hidden_states, out.logits


# --------------------------------------------------------------------------- #
# Subject-token hidden states (for probing + drift)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def subject_hidden_states(model, tokenizer, subj, prop):
    """Per-layer hidden state at the subject's last token. -> [L+1, H] float32.

    Returns None if the subject can't be located in the cloze prompt.
    """
    prompt = D.cloze_prompt(subj, prop)
    if prompt is None:
        return None
    idx = D.subject_last_token_index(prompt, subj, tokenizer)
    if idx is None:
        return None
    _, hs, _ = _hidden_states(model, tokenizer, prompt)
    vecs = torch.stack([h[0, idx] for h in hs]).float().cpu().numpy()
    return vecs           # [num_layers+1, hidden]


# --------------------------------------------------------------------------- #
# Logit lens: resolution depth + margin
# --------------------------------------------------------------------------- #
@torch.no_grad()
def logit_lens_features(model, tokenizer, subj, prop, obj):
    """First layer whose top-1 (at the answer position) is the gold token, plus
    the final-layer margin between the gold token and the runner-up."""
    prompt = D.cloze_prompt(subj, prop)
    if prompt is None:
        return {"resolution_depth": np.nan, "answer_margin": np.nan,
                "resolution_depth_frac": np.nan}
    gold_ids = D.answer_token_ids(obj, tokenizer)
    if not gold_ids:
        return {"resolution_depth": np.nan, "answer_margin": np.nan,
                "resolution_depth_frac": np.nan}
    gold0 = gold_ids[0]                    # first answer token

    _, hs, logits = _hidden_states(model, tokenizer, prompt)
    norm, head = _final_norm_and_head(model)
    pos = -1                               # position predicting the answer
    n_layers = len(hs)

    depth = np.nan
    for layer, h in enumerate(hs):
        vec = h[0, pos]
        if norm is not None:
            vec = norm(vec)
        layer_logits = head(vec)
        if int(layer_logits.argmax()) == gold0:
            depth = layer
            break

    # margin at the true final layer
    final_logits = logits[0, pos].float()
    gold_logit = final_logits[gold0].item()
    tmp = final_logits.clone()
    tmp[gold0] = -float("inf")
    runner_up = tmp.max().item()
    return {
        "resolution_depth": depth,
        "resolution_depth_frac": depth / (n_layers - 1) if not np.isnan(depth) else np.nan,
        "answer_margin": gold_logit - runner_up,
    }


# --------------------------------------------------------------------------- #
# Representation drift between consecutive checkpoints
# --------------------------------------------------------------------------- #
def representation_drift(vecs_t, vecs_tp1):
    """Cosine similarity per layer between subject hidden states at t and t+1.

    Low similarity = the subject was rewritten between checkpoints.  Feed
    ``subject_hidden_states`` output from two adjacent checkpoints.
    """
    if vecs_t is None or vecs_tp1 is None:
        return {"drift_cos_mean": np.nan, "drift_cos_last": np.nan}
    a = vecs_t / (np.linalg.norm(vecs_t, axis=-1, keepdims=True) + 1e-8)
    b = vecs_tp1 / (np.linalg.norm(vecs_tp1, axis=-1, keepdims=True) + 1e-8)
    cos = (a * b).sum(-1)                  # per layer
    return {"drift_cos_mean": float(cos.mean()),
            "drift_cos_last": float(cos[-1])}


# --------------------------------------------------------------------------- #
# Paraphrase stability
# --------------------------------------------------------------------------- #
@torch.no_grad()
def paraphrase_stability(model, tokenizer, row):
    """Run 2-3 phrasings; report gold-prob spread and correctness agreement.

    A fact that only works under one exact phrasing is likely a memorized
    surface pattern -- the guess is that those die first.
    """
    prompts = D.paraphrase_prompts(row["subj"], row["prop"], row["question"])
    answers = recall.acceptable_answers(row)
    probs, corrects = [], []
    for p in prompts:
        probs.append(recall.gold_answer_logprob(model, tokenizer, p, row["obj"]))
        corrects.append(recall.greedy_correct(model, tokenizer, p, answers))
    probs = np.array(probs, dtype=float)
    return {
        "paraphrase_n": len(prompts),
        "paraphrase_prob_mean": float(np.nanmean(probs)),
        "paraphrase_prob_std": float(np.nanstd(probs)),
        "paraphrase_correct_frac": float(np.mean(corrects)),
        "paraphrase_all_correct": bool(np.all(corrects)),
    }


# --------------------------------------------------------------------------- #
# Linear probe over collected subject hidden states (offline, CPU)
# --------------------------------------------------------------------------- #
def train_layer_probes(X_by_layer, y, n_splits=5, max_iter=1000):
    """Train a linear classifier per layer to predict the answer from the
    subject hidden state; return cross-val accuracy per layer.

    Parameters
    ----------
    X_by_layer : np.ndarray [n_facts, n_layers, hidden]
    y : array [n_facts] answer labels (e.g. obj_id, label-encoded)

    A high-accuracy layer means the fact is stored "cleanly" (linearly
    readable); facts that are correct but not readable may be more fragile.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    n_layers = X_by_layer.shape[1]
    accs = []
    for layer in range(n_layers):
        X = X_by_layer[:, layer, :]
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=max_iter, n_jobs=-1))
        acc = cross_val_score(clf, X, y, cv=n_splits, scoring="accuracy").mean()
        accs.append(float(acc))
    return np.array(accs)
