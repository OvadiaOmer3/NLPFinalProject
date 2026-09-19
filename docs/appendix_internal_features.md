# Appendix X — Internal-state feature extraction

This appendix documents the *internal* features: the quantities we read out of the
model's activations at every checkpoint, as opposed to the corpus-side (timeline,
competition, mention) and behavioural (recall trajectory) features. All of them are
produced by `src/features/internal.py::extract_fact`, driven by
`feature_extraction/eval_checkpoints.py`, and assembled into the modelling tables by
`legacy/build_feature_table.py` (the superseded panel/facts pipeline).

## X.1 Extraction protocol

For every (checkpoint, fact) pair the model is touched exactly twice:

1. **Greedy decode** on the evaluation prompt (15-shot cloze by default) → the
   generation and the substring-match label `correct` used throughout the paper.
2. **One teacher-forced forward pass** on `[prompt] ++ [gold answer tokens]` with
   `output_hidden_states=True` and a forward pre-hook on every `mlp.down_proj` →
   *all* internal features below, plus the optional hidden-state and neuron dumps.

Notation used below. Let the prompt have $n$ tokens; the **answer position** is
$t = n$, the last prompt token, i.e. the position whose next-token distribution *is*
the answer. Let the gold object tokenize (with a leading space) to
$g_1,\dots,g_m$. The model has $L$ transformer blocks, so `output_hidden_states`
returns $L+1$ residual states $h^{(0)},\dots,h^{(L)}$ ($h^{(0)}$ = embeddings,
$h^{(L)}$ = post-final-RMSNorm); for LMEnt-1B, $L=18$, $d_\text{model}=2048$,
$d_\text{ff}=5632$. Write $W_U$ for the unembedding (LM head) and
$z = W_U h^{(L)}_t \in \mathbb{R}^{V}$ for the final logits at the answer position,
with $p = \mathrm{softmax}(z)$.

Two prompt conventions matter:

* **Internals are read on the zero-shot cloze by default** (`--internal-fewshot`
  switches them to the few-shot prompt). The answer position is then unconditioned
  and therefore comparable across shot counts, even in a sweep whose `correct` was
  scored on a 15-shot prompt.
* The **subject's last token** is located by mapping the character span of the
  subject string to token indices via the fast tokenizer's offset mapping, using
  `rfind` (the fact's own cloze is always the *tail* of the prompt; under a few-shot
  prefix the subject string can also occur inside the demonstration block — 13 / 1452
  facts in the 15-shot top-decile run). `subject_found` records whether the lookup
  succeeded; all `*_subj` features are NaN when it did not.

## X.2 Output-distribution features (answer position, final layer)

These describe the single next-token distribution the model would sample the answer
from. They all read the **first** answer token $g_1$, so they describe one
distribution rather than a multi-token sequence; `gold_ntokens` $=m$ is kept so that
multi-token answers can be filtered downstream (e.g. treat `is_top1` as reliable only
when $m=1$).

| Feature | Definition |
|---|---|
| `gold_prob` | length-normalised teacher-forced probability of the whole answer, $\big(\prod_{j=1}^{m} p(g_j \mid x, g_{<j})\big)^{1/m} = \exp\!\big(\tfrac{1}{m}\sum_j \log p(g_j\mid x,g_{<j})\big)$ |
| `gold_prob_first` | $p_{g_1}$ — probability of the first answer token alone |
| `gold_surprisal` | $-\log p_{g_1}$, in nats |
| `top1_prob` | $\max_v p_v$ — how confident the model is in whatever it prefers |
| `pred_entropy` | $H(p) = -\sum_{v} p_v \log p_v$ — peakedness of the guess; high = the model is diffuse over many candidates |
| `is_top1` | $\mathbb{1}[\arg\max_v z_v = g_1]$ — internal correctness of the first token (cheaper and stricter than the decode-based `correct`) |
| `gold_rank` | $\lvert\{v : z_v > z_{g_1}\}\rvert$ — 0 iff gold is top-1 |
| `answer_margin` | $z_{g_1} - \max_{v \neq g_1} z_v$ — signed logit gap to the runner-up; $>0$ iff gold is top-1 |
| `logit_gap_top` | $\max_v z_v - z_{g_1} \ge 0$ — distance from the top logit; $0$ iff gold is top-1 |
| `gold_ntokens` | $m$, number of tokens the gold answer spans |

`answer_margin` and `logit_gap_top` are two views of the same gap, related by
$\texttt{logit\_gap\_top} = \max(0,\, -\texttt{answer\_margin})$: the margin resolves
*how comfortably* a correct fact wins, the gap resolves *how far off* an incorrect one
is. `gold_prob` reproduces the teacher-forced score used by the recall pass
(`recall.gold_answer_logprob`), so no separate forward is needed for it.

Facts whose answer does not tokenize (empty $g$) get NaN for all gold-dependent
columns, `gold_rank = -1` and `is_top1 = False`.

## X.3 Logit-lens depth features

We apply the unembedding to every intermediate residual state at the answer position
(*logit lens*; nostalgebraist 2020) to ask **where in the stack** the answer becomes
the model's preferred token:

$$
z^{(\ell)} \;=\;
\begin{cases}
W_U\,\mathrm{RMSNorm}\!\left(h^{(\ell)}_t\right), & 0 \le \ell < L\\[2pt]
W_U\,h^{(\ell)}_t, & \ell = L
\end{cases}
$$

The $\ell = L$ case skips the norm on purpose: `output_hidden_states` returns the last
entry *already normalised*, so $W_U h^{(L)}$ reproduces the model's true logits
exactly, whereas norming it a second time would rescale the residual by the learned
RMSNorm weight twice and can change the argmax — corrupting the one boundary where the
lens is not an approximation.

| Feature | Definition |
|---|---|
| `resolution_depth` | $d = \min\{\ell : \arg\max_v z^{(\ell)}_v = g_1\}$, NaN if the gold token is never top-1 at any depth |
| `resolution_depth_frac` | $d / L$ — depth normalised to $[0,1]$, comparable across models |
| `gold_rank_by_layer` | $\big(\lvert\{v: z^{(\ell)}_v > z^{(\ell)}_{g_1}\}\rvert\big)_{\ell=0}^{L}$ — the gold token's rank trajectory through the stack |
| `gold_logit_by_layer` | $\big(z^{(\ell)}_{g_1}\big)_{\ell=0}^{L}$ — its raw logit trajectory |

The two `*_by_layer` columns are lists and are kept only in the raw per-checkpoint
recall parquets (depth-trajectory analyses); the flat modelling table carries the two
scalars. `gold_rank_by_layer` is what the mid-stack analysis (§8) is computed from.

## X.4 Residual-stream magnitude

$$\texttt{resid\_norm\_by\_layer}[\ell] \;=\; \lVert h^{(\ell)}_t \rVert_2,
\qquad \texttt{resid\_norm\_last} \;=\; \lVert h^{(L)}_t \rVert_2 .$$

A scale control for the lens features: logit differences grow with the residual norm,
so a change in `answer_margin` is only interesting if the norm did not simply grow.

## X.5 MLP activity features

The forward pre-hook on `mlp.down_proj` captures its input, which for the OLMo2 SwiGLU
MLP is the vector of actual per-neuron activations of the feed-forward layer:

$$a^{(\ell)} \;=\; \mathrm{SiLU}\!\left(W^{(\ell)}_\text{gate} x\right) \odot \left(W^{(\ell)}_\text{up} x\right) \;\in\; \mathbb{R}^{d_\text{ff}} .$$

They are captured at **two** positions only (to stay cheap): the answer position
(where the object is formed) and the subject's last token (where the entity itself is
represented). Answer-position columns are unsuffixed; subject-position columns carry
the `_subj` suffix.

| Feature | Definition |
|---|---|
| `mlp_active_frac_by_layer` | $\dfrac{1}{d_\text{ff}}\sum_{i} \mathbb{1}\big[\,\lvert a^{(\ell)}_i\rvert > \varepsilon\,\big]$, with $\varepsilon = 10^{-3}$ — fraction of neurons "on" |
| `mlp_l1_by_layer` | $\dfrac{1}{d_\text{ff}}\sum_i \lvert a^{(\ell)}_i \rvert$ — mean activation magnitude (overall drive level) |
| `mlp_participation_by_layer` | $\mathrm{PR}(a^{(\ell)}) = \dfrac{\big(\sum_i (a^{(\ell)}_i)^2\big)^2}{\sum_i (a^{(\ell)}_i)^4}$ |
| `mlp_active_frac_mean`, `mlp_participation_mean` | layer-averaged (nanmean over $\ell$) scalars carried into the flat tables, each with a `_subj` variant |

SiLU never emits an exact zero, so "off" means *small*, not literally 0 — hence the
hard threshold $\varepsilon$. The **participation ratio** $\mathrm{PR}$ is the
*effective number of neurons* carrying the activation energy: it is invariant to
rescaling $a \mapsto ca$, equals 1 when a single neuron dominates and $d_\text{ff}$
when the energy is spread uniformly. It complements the threshold-based active
fraction, which is sensitive to the overall drive level. Measured on our sweep,
$\mathrm{PR}$ has median $\approx 8$ and 99.9th percentile $\approx 58$ at both
positions ($d_\text{ff} = 5632$).

## X.6 Dumped arrays (not table columns)

Two optional streaming dumps are written per checkpoint, memmapped so the full tensor
never has to be held in RAM. Row $i$ corresponds to `ids[i]` in the sibling
`*_ids.npy`; positions that could not be located stay NaN / sentinel.

* **Hidden states** — `hidden_step{S}_{pos}.npy`, `float16`,
  $[\,n_\text{facts},\,L{+}1,\,d_\text{model}\,]$ per position: the full residual
  trajectory of the answer token and of the subject's last token. Values are clamped
  to $\pm 65504$ before the bf16→f16 cast, since late-layer residuals occasionally
  exceed the f16 range and would otherwise become silent infinities. ≈76 KB per fact
  per checkpoint per position. Feeds the offline probes and representation drift.
* **Top-$k$ active neurons** — `neurons_step{S}_{pos}_idx.npy` (`uint16`) and
  `_val.npy` (`float16`), both $[\,n_\text{facts},\,L,\,k\,]$: the indices of the $k$
  most active MLP neurons per layer, ordered by descending $\lvert a_i \rvert$, and
  their **signed** activations. Sign is retained deliberately — two facts driving the
  same neuron in opposite directions are not agreeing, so magnitude-only values would
  overstate overlap. Missing rows carry the sentinel `65535` (never a valid index,
  since $d_\text{ff} \ll 65535$) and NaN. We use $k = 128$: with $\mathrm{PR}$'s
  99.9th percentile at $\approx 58$ this covers the tail with 2× headroom while
  staying clear of the several hundred generically-active neurons per mid-stack layer,
  at ≈6 % of the hidden dump's size. Indices support set-overlap / circuit-churn
  analyses; the values additionally support magnitude-weighted overlap.

## X.7 How internal features enter the tables

* **`panel.parquet`** (one row per fact × checkpoint) carries every internal scalar
  *at that checkpoint*, alongside `correct`, `gold_prob`, and the time-varying timeline
  and competition features. This is the table all trajectory analyses use.
* **`facts.parquet`** (one row per fact, the retention-prediction table) carries the
  internal state at the **first checkpoint at which the fact was learned**, prefixed
  `atlearn_*`, and NaN for never-learned facts.

The choice of the first-learned checkpoint is a leakage constraint, not a convenience:
the prediction target `retained` is *defined by* correctness at the final checkpoint,
so internal state measured at the final checkpoint would trivially encode the label
(the same reason the `leaky_*` columns are excluded). State at acquisition time —
"how cleanly was this fact stored when it first arrived" — is a legitimate predictor of
whether it will stick.

## X.8 Implemented but not used in the reported experiments

`src/features/internal.py` also ships three probes that are not part of the reported
results, listed here for completeness:

* **Representation drift** — per-layer cosine similarity of the subject's hidden state
  between consecutive checkpoints,
  $\cos^{(\ell)} = \frac{\langle h^{(\ell)}_{s}, h^{(\ell)}_{s+1}\rangle}{\lVert h^{(\ell)}_s\rVert\,\lVert h^{(\ell)}_{s+1}\rVert}$
  for checkpoints $s, s{+}1$, summarised as its mean over layers and its last-layer
  value. Low similarity = the subject representation was rewritten between checkpoints.
* **Paraphrase stability** — the same fact under the template cloze plus 1–2 variants
  derived from the PopQA question; reports the mean and standard deviation of
  `gold_prob` across phrasings and the fraction of phrasings answered correctly. A fact
  that only survives one exact phrasing is likely a memorised surface pattern.
* **Per-layer linear probes** — a cross-validated logistic classifier per layer
  predicting the answer from the subject's hidden state; high accuracy at a layer means
  the fact is linearly readable ("cleanly stored") there.
