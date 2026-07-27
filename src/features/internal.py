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

Prompts are single-sourced from olmes (``src.lment_popqa`` via ``src.data``):
the zero-shot cloze is used for probe / logit lens / hidden states so the answer
position is unconditioned.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src import data as D
from src import lment_popqa
from src import recall

# float16 can represent magnitudes up to this; late-layer residuals occasionally
# exceed it, so we clamp before the bf16 -> f16 cast to avoid silent infs.
_F16_MAX = 65504.0


# --------------------------------------------------------------------------- #
# Model internals accessors (HF Olmo2ForCausalLM)
# --------------------------------------------------------------------------- #
def _final_norm_and_head(model):
    """Final RMSNorm + LM head for the OLMo2 architecture LMEnt ships.

    For ``transformers`` ``Olmo2ForCausalLM`` these are ``model.model.norm`` and
    ``model.lm_head`` (kept as a getattr with a tied-embeddings fallback for the
    head).  VERIFY once against ``print(model)`` on the real checkpoint.
    """
    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None)
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
def subject_hidden_states(model, tokenizer, row):
    """Per-layer hidden state at the subject's last token. -> [L+1, H] float32.

    Returns None if the subject can't be located in the cloze prompt.
    """
    prompt = D.zero_shot_cloze(row)
    idx = D.subject_last_token_index(prompt, row["subj"], tokenizer)
    if idx is None:
        return None
    _, hs, _ = _hidden_states(model, tokenizer, prompt)
    vecs = torch.stack([h[0, idx] for h in hs]).float().cpu().numpy()
    return vecs           # [num_layers+1, hidden]


# --------------------------------------------------------------------------- #
# Logit lens: resolution depth + margin
# --------------------------------------------------------------------------- #
@torch.no_grad()
def logit_lens_features(model, tokenizer, row):
    """First layer whose top-1 (at the answer position) is the gold token, plus
    the final-layer margin between the gold token and the runner-up."""
    prompt = D.zero_shot_cloze(row)
    gold_ids = D.answer_token_ids(row["obj"], tokenizer)
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
def paraphrase_prompts(row) -> list[str]:
    """A small set of surface phrasings of the same fact for the stability test:
    the olmes template cloze plus 1-2 natural-language variants from the PopQA
    question.  (olmes ships one template per relation, so the alternates come
    from the dataset's question field.)"""
    prompts = [D.zero_shot_cloze(row)]
    q = row.get("question")
    if isinstance(q, str) and q:
        prompts.append(q.rstrip(" ?") + " is")    # declarative variant
        prompts.append(q if q.endswith(" ") else q + " ")
    return prompts


@torch.no_grad()
def paraphrase_stability(model, tokenizer, row):
    """Run the phrasings; report gold-prob spread and correctness agreement.

    A fact that only works under one exact phrasing is likely a memorized
    surface pattern -- the guess is that those die first.
    """
    prompts = paraphrase_prompts(row)
    probs, corrects = [], []
    for p in prompts:
        probs.append(recall.gold_answer_logprob(model, tokenizer, p, row["obj"]))
        gen = recall.greedy_generation(model, tokenizer, p)
        corrects.append(lment_popqa.is_correct(row, gen))
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


# --------------------------------------------------------------------------- #
# One-pass extractor: every internal signal from a single forward
# --------------------------------------------------------------------------- #
# The functions above each run their own forward pass (handy in a notebook on a
# single fact).  For the full sweep we instead want *one* forward per (checkpoint,
# fact): teacher-force the gold answer after the zero-shot cloze so a single call
# yields recall probability, next-token geometry, logit-lens depth, per-layer
# residual / MLP activity, and the f16 hidden-state dump all at once.  The only
# thing it can't give is `correct` (that needs a greedy multi-token decode).

def _decoder_layers(model):
    """The list of transformer blocks (Olmo2Model.layers)."""
    base = getattr(model, "model", model)
    return base.layers


@torch.no_grad()
def extract_fact(model, tokenizer, row, *, positions=("answer", "subject_last"),
                 neuron_eps=1e-3, per_layer=True, top_k=None):
    """Run the model ONCE on ``[zero-shot cloze] + [gold answer tokens]`` and
    return ``(features, hidden, neurons)``.

    ``features`` mixes scalars (safe to put straight in a parquet) with a few
    per-layer lists (only when ``per_layer=True``).  ``hidden`` maps each name in
    ``positions`` to an ``[L+1, H] float16`` stack of that token's residual state
    across every layer (``None`` if the position couldn't be located).  ``neurons``
    is ``{}`` unless ``top_k`` is set, in which case it maps ``answer`` /
    ``subject_last`` to a ``[n_layers, k] uint16`` array of the indices of the k
    most active MLP neurons per layer (``None`` if that position wasn't located).

    The answer position is the prompt's last token -- the one whose next-token
    prediction *is* the answer -- so all logit-based features read that slot.
    """
    device = model.device
    prompt = D.zero_shot_cloze(row)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    n_prompt = prompt_ids.shape[1]
    ans_pos = n_prompt - 1                      # predicts the first answer token
    subj_idx = D.subject_last_token_index(prompt, row["subj"], tokenizer)

    gold_ids = D.answer_token_ids(row["obj"], tokenizer)
    if gold_ids:
        ans_t = torch.tensor([gold_ids], device=device)
        full = torch.cat([prompt_ids, ans_t], dim=1)
    else:
        full = prompt_ids

    # How many tokens the gold answer spans.  The logit-geometry features below
    # (logit_gap_top / gold_rank / is_top1 / answer_margin) look at the FIRST
    # answer token only -- they describe the single next-token distribution at
    # the answer position.  Only `gold_prob` is length-normalized over all
    # tokens.  Keep this so downstream can filter multi-token answers (e.g. treat
    # `is_top1` as reliable only when gold_ntokens == 1).
    n_gold_tokens = len(gold_ids)

    # --- hooks: capture the SwiGLU intermediate (down_proj input) = "neurons" --
    # For Olmo2MLP the input to down_proj is act_fn(gate(x)) * up(x): the actual
    # per-neuron activations of the feed-forward layer.  We grab it at both the
    # answer position (where the object is formed) and the subject's last token
    # (where the entity itself is represented) -- only those slots, to stay cheap.
    neuron_pos = {"answer": ans_pos}
    if subj_idx is not None:
        neuron_pos["subject_last"] = subj_idx
    layers = _decoder_layers(model)
    mlp_act = [None] * len(layers)          # each -> {pos_name: [intermediate] tensor}
    handles = []

    def _mk(i):
        def hook(_mod, inp):
            row_act = inp[0][0]             # [T, intermediate]
            mlp_act[i] = {name: row_act[idx].detach().float()
                          for name, idx in neuron_pos.items()}
        return hook

    for i, lyr in enumerate(layers):
        handles.append(lyr.mlp.down_proj.register_forward_pre_hook(_mk(i)))
    try:
        out = model(full, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()

    hs = out.hidden_states                      # (L+1) tuple of [1, T, H]
    logits = out.logits                         # [1, T, V]
    n_hs = len(hs)
    norm, head = _final_norm_and_head(model)
    feats: dict = {"subject_found": subj_idx is not None,
                   "gold_ntokens": n_gold_tokens}

    # ---- next-token geometry at the answer position (true final layer) -------
    final_logits = logits[0, ans_pos].float()
    logp = torch.log_softmax(final_logits, dim=-1)
    p = logp.exp()
    top1 = int(final_logits.argmax())
    max_logit = float(final_logits[top1])
    feats["pred_entropy"] = float(-(p * logp).sum())    # nats; peakedness of the guess
    feats["top1_prob"] = float(p[top1])

    gold0 = gold_ids[0] if gold_ids else None
    if gold0 is not None:
        gold_logit = float(final_logits[gold0])
        tmp = final_logits.clone()
        tmp[gold0] = -float("inf")
        runner_up = float(tmp.max())
        feats["gold_prob_first"] = float(p[gold0])
        feats["gold_surprisal"] = float(-logp[gold0])          # nats
        feats["answer_margin"] = gold_logit - runner_up        # >0 iff gold is top-1
        feats["logit_gap_top"] = max_logit - gold_logit        # "distance" from the top logit; 0 iff top-1
        feats["is_top1"] = bool(top1 == gold0)
        feats["gold_rank"] = int((final_logits > gold_logit).sum())  # 0 == top-1
        # length-normalized teacher-forced gold prob (== recall.gold_answer_logprob)
        if full.shape[1] > n_prompt:
            span = torch.log_softmax(
                logits[0, ans_pos:ans_pos + len(gold_ids)].float(), dim=-1)
            tok_lp = span[torch.arange(len(gold_ids)),
                          torch.tensor(gold_ids, device=span.device)]
            feats["gold_prob"] = float(np.exp(tok_lp.mean().item()))
        else:
            feats["gold_prob"] = feats["gold_prob_first"]
    else:
        for k in ("gold_prob", "gold_prob_first", "gold_surprisal",
                  "answer_margin", "logit_gap_top"):
            feats[k] = float("nan")
        feats["gold_rank"] = -1
        feats["is_top1"] = False

    # ---- logit lens over depth + residual growth at the answer position ------
    depth = np.nan
    gold_rank_by_layer, gold_logit_by_layer, resid_norm_by_layer = [], [], []
    for layer, h in enumerate(hs):
        vec = h[0, ans_pos]
        resid_norm_by_layer.append(float(vec.float().norm()))
        lens_vec = norm(vec) if norm is not None else vec
        layer_logits = head(lens_vec).float()
        if gold0 is not None:
            gl = float(layer_logits[gold0])
            gold_logit_by_layer.append(gl)
            gold_rank_by_layer.append(int((layer_logits > gl).sum()))
            if np.isnan(depth) and int(layer_logits.argmax()) == gold0:
                depth = layer
    feats["resolution_depth"] = depth
    feats["resolution_depth_frac"] = (depth / (n_hs - 1)
                                      if not np.isnan(depth) else np.nan)
    feats["resid_norm_last"] = resid_norm_by_layer[-1]

    # ---- MLP active-neuron activity per layer, at each position (from hooks) --
    # "answer" columns are unsuffixed (back-compat); "subject_last" gets _subj.
    # A neuron counts as active when |activation| > neuron_eps (SwiGLU's SiLU
    # never emits an exact zero, so "off" means small, not literally 0).  The
    # participation ratio (Sum a^2)^2 / Sum a^4 is the *effective number* of neurons
    # carrying the activation energy -- a scale-free concentration measure (near 1
    # = a few neurons dominate, high = spread out) that complements the hard-
    # threshold active fraction.
    neurons: dict = {}
    for pos_name, suffix in (("answer", ""), ("subject_last", "_subj")):
        fracs, l1s, prs, topk_idx = [], [], [], []
        for a in mlp_act:
            vec = a.get(pos_name) if a is not None else None
            if vec is None:
                fracs.append(np.nan)
                l1s.append(np.nan)
                prs.append(np.nan)
                topk_idx.append(None)
                continue
            absv = vec.abs()
            fracs.append(float((absv > neuron_eps).float().mean()))
            l1s.append(float(absv.mean()))
            energy = vec.pow(2)
            denom = float(energy.pow(2).sum())               # Sum a^4
            prs.append(float(energy.sum().item() ** 2 / denom) if denom > 0 else np.nan)
            if top_k:
                kk = min(int(top_k), absv.numel())
                topk_idx.append(torch.topk(absv, kk).indices.cpu().numpy().astype(np.uint16))
        feats[f"mlp_active_frac_mean{suffix}"] = (
            float(np.nanmean(fracs)) if fracs and not np.all(np.isnan(fracs)) else np.nan)
        feats[f"mlp_participation_mean{suffix}"] = (
            float(np.nanmean(prs)) if prs and not np.all(np.isnan(prs)) else np.nan)
        if per_layer:
            feats[f"mlp_active_frac_by_layer{suffix}"] = fracs
            feats[f"mlp_l1_by_layer{suffix}"] = l1s
            feats[f"mlp_participation_by_layer{suffix}"] = prs
        if top_k:
            neurons[pos_name] = (np.stack(topk_idx)
                                 if all(x is not None for x in topk_idx) else None)

    if per_layer:
        feats["gold_rank_by_layer"] = gold_rank_by_layer or None
        feats["gold_logit_by_layer"] = gold_logit_by_layer or None
        feats["resid_norm_by_layer"] = resid_norm_by_layer

    # ---- f16 hidden-state dump at the requested positions --------------------
    pos_idx = {"answer": ans_pos, "subject_last": subj_idx}
    hidden: dict = {}
    for name in positions:
        idx = pos_idx.get(name)
        if idx is None:
            hidden[name] = None
            continue
        stack = torch.stack([h[0, idx] for h in hs]).float()
        stack = stack.clamp_(-_F16_MAX, _F16_MAX).to(torch.float16)
        hidden[name] = stack.cpu().numpy()          # [L+1, H] float16
    return feats, hidden, neurons


# --------------------------------------------------------------------------- #
# f16 hidden-state store (one memmap per position, per checkpoint)
# --------------------------------------------------------------------------- #
class HiddenDump:
    """Streaming writer for the per-checkpoint hidden-state dump.

    Backs each requested position with its own ``float16`` ``.npy`` memmap of
    shape ``[n_facts, n_layers+1, hidden]`` so we never hold the whole tensor in
    RAM.  Row ``i`` corresponds to ``ids[i]``; positions that couldn't be located
    for a fact are left as NaN.  Reload later with ``np.load(path, mmap_mode='r')``.
    """

    def __init__(self, step, ids, n_layers_plus1, hidden_size, positions, out_dir,
                 resume=False):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.step = step
        n = len(ids)
        self.arrs = {}
        self.paths = {}
        for name in positions:
            path = out_dir / f"hidden_step{step}_{name}.npy"
            shape = (n, n_layers_plus1, hidden_size)
            if resume and path.exists():
                # Reopen in place so rows written before a crash / preemption
                # survive; w+ would truncate them back to NaN.
                arr = np.lib.format.open_memmap(path, mode="r+")
                if arr.shape != shape:
                    raise ValueError(
                        f"{path.name} has shape {arr.shape}, expected {shape}; "
                        "delete the partial artifacts for this step and rerun.")
            else:
                arr = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float16, shape=shape)
                arr[:] = np.nan
            self.arrs[name] = arr
            self.paths[name] = path
        np.save(out_dir / f"hidden_step{step}_ids.npy", np.asarray(ids))
        (out_dir / f"hidden_step{step}_meta.json").write_text(json.dumps({
            "step": int(step), "n_facts": int(n),
            "n_layers_plus1": int(n_layers_plus1),
            "hidden_size": int(hidden_size),
            "positions": list(positions), "dtype": "float16",
        }, indent=2))

    def write(self, i, hidden):
        for name, arr in self.arrs.items():
            v = hidden.get(name)
            if v is not None:
                arr[i] = v

    def close(self):
        for arr in self.arrs.values():
            arr.flush()


# --------------------------------------------------------------------------- #
# Top-k active-neuron index store (one memmap per position, per checkpoint)
# --------------------------------------------------------------------------- #
class NeuronDump:
    """Streaming writer for the top-k active MLP-neuron *indices* per (fact, layer).

    One ``uint16`` ``.npy`` memmap per position, shape ``[n_facts, n_layers, k]``,
    holding for each fact the indices of the ``k`` most active feed-forward
    neurons in every layer (ordered by descending ``|activation|``).  Row ``i``
    corresponds to ``ids[i]``; rows for a position that couldn't be located are
    filled with the sentinel ``MISSING`` (65535, never a valid neuron index since
    ``intermediate_size`` << 65535).  Indices alone are enough for the overlap /
    churn analyses (which neurons a fact reuses across checkpoints or shares with
    other facts) at ~3% of the hidden-state dump's size.
    """

    MISSING = int(np.iinfo(np.uint16).max)      # 65535

    def __init__(self, step, ids, n_layers, k, positions, out_dir, resume=False):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.step = step
        self.k = int(k)
        n = len(ids)
        self.arrs = {}
        self.paths = {}
        for name in positions:
            path = out_dir / f"neurons_step{step}_{name}.npy"
            shape = (n, n_layers, int(k))
            if resume and path.exists():
                arr = np.lib.format.open_memmap(path, mode="r+")
                if arr.shape != shape:
                    raise ValueError(
                        f"{path.name} has shape {arr.shape}, expected {shape}; "
                        "delete the partial artifacts for this step and rerun.")
            else:
                arr = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.uint16, shape=shape)
                arr[:] = self.MISSING
            self.arrs[name] = arr
            self.paths[name] = path
        np.save(out_dir / f"neurons_step{step}_ids.npy", np.asarray(ids))
        (out_dir / f"neurons_step{step}_meta.json").write_text(json.dumps({
            "step": int(step), "n_facts": int(n), "n_layers": int(n_layers),
            "k": int(k), "positions": list(positions), "dtype": "uint16",
            "missing_sentinel": self.MISSING,
            "order": "descending |activation|",
        }, indent=2))

    def write(self, i, neurons):
        for name, arr in self.arrs.items():
            v = neurons.get(name)
            if v is not None:                    # [n_layers, k] uint16
                arr[i] = v

    def close(self):
        for arr in self.arrs.values():
            arr.flush()
