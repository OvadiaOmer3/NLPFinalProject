# Features used in `02_internal_paper_figures.ipynb`

A reference for every quantity the notebook reads, derives, plots, or feeds to a
classifier. It is a companion to [`appendix_internal_features.md`](appendix_internal_features.md),
which documents the *extractor* (`src/features/internal.py`); this file documents the
*consumer* — which of those columns the two paper figures actually use, and what the
notebook builds on top of them.

**Scope.** The notebook produces Fig. 1 (logit-lens depth profile) and Fig. 2
(neuron churn + predictive AUC, with a 2×2 supplement). It reads two of the four inputs
its header table lists: the flat recall parquets (input 1) and the top-128 neuron dump
(input 3). The fact-properties CSV (input 2) is loaded but never consumed by a figure,
and the hidden-state dump (input 4, Fig. 3a) is not touched at all — see
[§7](#7-loaded-but-not-used).

Throughout: $L = 18$ transformer blocks, so **19 layer boundaries** ($\ell = 0$
embeddings … $\ell = 18$ final) for the logit lens, and **18 MLP layers** for the neuron
features. $d_\text{ff} = 5632$, $k = 128$, 13 checkpoints, 1 310 scoreable facts.

---

## 1. Panel axes, labels, and grouping variables

Not "features" in the modelling sense, but everything downstream is indexed by them.

| Name | Source | What it is |
|---|---|---|
| `id` | parquet | fact identifier; the row axis of every panel, and the **group** for `GroupKFold` and every bootstrap |
| `step` | filename `recall_step{S}_…` | checkpoint (50 k … 658 032); the column axis of every panel |
| `correct` | parquet | olmes PopQACloze exact-match on a **greedy 15-shot** decode. The label, and the only quantity in the notebook measured on the few-shot prompt |
| `excluded` | parquet | olmes drops facts whose answer already appears in the query. A property of the *fact*, so filtering it leaves the panel balanced across checkpoints (1 452 → 1 310 rows) |

### Derived label and index arrays

| Name | Definition |
|---|---|
| `correct` (matrix) | `P["correct"]` as a boolean `(n_facts, 13)` array |
| `XF, XT, XFG` | the **transition panel**: one row per (fact $i$, checkpoint $t$) with $C_{i,t} = 1$. `XF` = fact row, `XT` = checkpoint index, `XFG` = $\lnot C_{i,t+1}$ — *forgotten at the next checkpoint*, the prediction target |
| `AF, AT, AFG` | the same, restricted to $t \ge 1$ — the **ante-hoc panel**, so a strictly backward churn window $t{-}1 \rightarrow t$ exists. 3 965 rows from 701 facts |
| `_was` | $C_{i,t-1}$ — *prior state*: was the fact already correct one checkpoint back. Used as its own predictor (`prior state alone`) and as the stratifier of the Fig. 2a 2×2 supplement |

A fact recurs once per checkpoint it survives, so transitions are **not** independent:
every interval in the notebook is a cluster bootstrap over facts, and every
cross-validation is `GroupKFold(groups=fact)`.

---

## 2. Output-distribution scalars (answer position, final layer)

Read from the recall parquets and pivoted into `P[feature]` of shape `(n_facts, 13)`.
All are measured on the **zero-shot cloze** (the shared prompt caveat) at the first gold
token $g_1$, with $z$ the final logits and $p = \mathrm{softmax}(z)$.

| Feature | Definition | Used by |
|---|---|---|
| `gold_prob` | $\exp\!\big(\tfrac{1}{m}\sum_j \log p(g_j \mid x, g_{<j})\big)$ — length-normalised probability of the whole gold answer | Fig. 2a tercile control, Fig. 2b (both blocks), 2×2 supplement |
| `gold_rank` | $\lvert\{v : z_v > z_{g_1}\}\rvert$, **0-based** (0 = top-1) | Fig. 2b; also repairs boundary 18 of the lens (§3) |
| `answer_margin` | $z_{g_1} - \max_{v \neq g_1} z_v$ — signed logit gap to the runner-up, $> 0$ iff gold is top-1 | Fig. 2b |
| `pred_entropy` | $H(p) = -\sum_v p_v \log p_v$, nats — the model's diffuseness over the whole vocabulary. **The one feature that never looks at the gold answer** | Fig. 2b |
| `is_top1`, `resolution_depth`, `top1_prob` | loaded into `P` / `NEWP`; see [§7](#7-loaded-but-not-used) | — |

### Sign orientation

Every scalar entering a classifier is flipped so that **higher = more likely to be
forgotten**, which makes the fitted coefficients readable:

```
"gold probability":   -gold_prob
"gold rank":           log10(gold_rank + 1)      # log, because rank spans 0 … |V|
"answer margin":      -answer_margin
"predictive entropy":  pred_entropy              # already oriented
```

These four are the block labelled **"internal scalar features"** — the honest baseline
Fig. 2b has to beat, i.e. everything the output distribution already says.

---

## 3. The logit lens: `gold_rank_by_layer` → `GRL`

The per-layer array is the feature that turns Fig. 1 from a tautology into a
localisation. Stacked over checkpoints into

$$\texttt{GRL}[i, t, \ell] \;=\; \big\lvert\{v : z^{(\ell)}_v > z^{(\ell)}_{g_1}\}\big\rvert,
\qquad \text{shape } (1310,\ 13,\ 19).$$

**Boundary 18 is repaired inside the notebook.** `output_hidden_states` returns the last
entry already post-final-RMSNorm, but the extractor applied the final norm a second time
before the LM head, so the stored value at $\ell = 18$ came from a doubly-normalised
state and agreed with the model's true output rank on only a minority of cells. The
notebook overwrites that column with the true final-layer `gold_rank`. Boundaries 0–17
are genuinely pre-norm and are left untouched.

### Transforms and aggregations built on `GRL`

| Quantity | Definition |
|---|---|
| `log_rank` | $\log_{10}(\texttt{GRL} + 1)$ — the $+1$ maps rank 0 (top-1) to 0; ranks span five orders of magnitude, so the log scale is what makes a median comparable |
| `cur` | `log_rank[XF, XT]` — one row per transition, 19 boundaries wide |
| `med_s`, `med_f` | **checkpoint-balanced** group medians: the median is taken *inside* each checkpoint, then averaged with equal weight. Pooling would let early checkpoints dominate the forgotten group and late ones the survivors |
| `gap` | `med_f - med_s`, with a 95 % interval from 500 cluster-bootstrap resamples over facts |
| `MID = slice(8, 14)` | the mid-stack band the gap localises the failure to. **Every later band average reuses this constant**, so Fig. 2 is measured on the region Fig. 1 identified |

---

## 4. Neuron features (top-128 dump)

`neuron_summaries(pos)` makes one streaming pass over the 13 checkpoints, producing
per-(fact, checkpoint, MLP layer) summaries from the top-$k$ **signed** activations $a$
and their indices $I$, at two positions: `answer` (→ `NA`) and `subject_last` (→ `NS`).
Let $\tilde p_j = \lvert a_j \rvert / \sum_j \lvert a_j \rvert$ over the top-$k$.

### 4a. State at one checkpoint — shape `(n_facts, 13, 18)`

| Key | Notebook label | Definition | Reads as |
|---|---|---|---|
| `H` | "high activation entropy" | $-\sum_j \tilde p_j \log \tilde p_j \,/\, \log k$ | 0 = one neuron carries everything, 1 = the top-$k$ are equal. Spread of the activation mass |
| `PR` | "diffuse activation" | $\big(\sum_j \lvert a_j \rvert\big)^2 \big/ \sum_j a_j^2$ | effective neuron count. **Note:** this is the top-$k$ $\ell_1/\ell_2$ form, *not* the extractor's $(\sum a^2)^2/\sum a^4$ (`mlp_participation_by_layer`) — different statistics, not numerically comparable |
| `L1` | "large activation mass" | $\sum_j \lvert a_j \rvert$ over the top-$k$ | total drive. Again a top-$k$ **sum**, unlike the extractor's `mlp_l1_by_layer` (a mean over all $d_\text{ff}$) |
| `GEN` | "private / shared neurons" | for each of the fact's $k$ neurons, the share of the **other** $n - 1$ panel facts that also have it in their top-$k$ at that layer, averaged over the $k$ | genericity. High = this fact runs on the layer's generic machinery; low = it uses private neurons |

### 4b. Change between consecutive checkpoints — shape `(n_facts, 12, 18)`

Both come from `_pair`, which concatenates and sorts the two index sets so a shared
neuron's two copies land side by side — one pass gives both the intersection size and
the dot product, with no $k \times k$ comparison and no dense 5632-wide buffer.

| Key | Definition | Reads as |
|---|---|---|
| `COS` | cosine similarity of the two sparse signed activation vectors over the top-$k$ support, $t \rightarrow t{+}1$ | 1 = the same neurons fire the same way; low = the circuit moved. Sign matters: two facts driving a neuron in opposite directions are not agreeing |
| `J32` | Jaccard of the **top-32 index sets**, $\lvert A \cap B \rvert / (64 - \lvert A \cap B \rvert)$ | a set-only, magnitude-blind version of the same question — a robustness check on `COS` |

### 4c. `within_z` — the standardisation that makes churn a fact-level statistic

Every churn number is z-scored **within its (interval, layer) column**:

$$\texttt{Z}[i, t, \ell] \;=\; \frac{M[i, t, \ell] - \mu_{t,\ell}}{\sigma_{t,\ell}}.$$

Consecutive checkpoints are 50 k optimizer steps apart and the whole network moves in
between, so raw similarity is dominated by *which interval it is*. Standardising within
the interval asks the only question that is about the fact: **did this fact's neurons
move more than its peers' did over the same 50 k steps?**

This produces `ZA_COS`, `ZA_J32` (answer position) and `ZS_COS` (subject position). Note
the orientation that results: **higher z = the fact's neurons moved *less***; the model
inputs negate it so that higher = more churn.

### 4d. Two churn windows

The distinction Fig. 2a rests on. `COS[:, t]` spans checkpoints $t \rightarrow t{+}1$, so:

| Window | Indexing | Status |
|---|---|---|
| across the flip | `Z[XF, XT]` (`lag_back=0`) | **descriptive** — the window contains the forgetting event, so predicting the flip from it is partly circular |
| before the flip | `Z[AF, AT - 1]` (`lag_back=1`) | **strictly ante-hoc** — the interval ended at the still-correct checkpoint; nothing in it can know about $t{+}1$ |

Everything predictive (Fig. 2b, the 2×2 supplement, the reported AUCs) uses the backward
window only. `bwd = nanmean(ZA_COS[AF, AT-1][:, MID], axis=1)` — mid-stack backward
churn — is the single scalar the supplement is built around.

---

## 5. The Fig. 2b feature blocks

Ten neuron scalars, all measured **at or before $t$**, each a band average over MLP
layers, oriented so higher = more likely forgotten. `band(D, key, sl, back)` is
`nanmean(D[key][AF, AT-back][:, sl], axis=1)`.

| Notebook name | Expression | Band (MLP layers) |
|---|---|---|
| churn, answer (mid) | $-\overline{\texttt{ZA\_COS}}$ | 8–13 |
| churn, answer (early) | $-\overline{\texttt{ZA\_COS}}$ | 0–7 |
| churn, answer (late) | $-\overline{\texttt{ZA\_COS}}$ | 14–17 |
| churn, subject (mid) | $-\overline{\texttt{ZS\_COS}}$ | 8–13 |
| top-32 churn (mid) | $-\overline{\texttt{ZA\_J32}}$ | 8–13 |
| diffuse activation (mid) | $+\overline{\texttt{PR}}$ | 8–13 |
| high activation entropy | $+\overline{\texttt{H}}$ | 8–13 |
| private neurons (mid) | $-\overline{\texttt{GEN}}$ | 8–13 |
| shared neurons (late) | $+\overline{\texttt{GEN}}$ | 12–15 |
| large activation mass | $\log_{10}\max(\overline{\texttt{L1}},\, 1)$ | 8–13 |

The five churn rows use the **backward** window (`AT - 1`); the five state rows are read
**at** $t$ (`back=0`) and are *not* z-scored — only `COS` / `J32` are.

### The four blocks compared

| Block | Columns |
|---|---|
| gold probability alone | 1 — the crudest "weakly stored" story |
| internal scalar features | the 4 output scalars of §2 |
| neuron features alone | the 10 above |
| internal scalar + neuron features | all 14 |

**How they are scored.** Logistic regression in a `SimpleImputer(median) →
StandardScaler → LogisticRegression` pipeline, fitted *inside* each training fold (no
global leakage), out-of-fold under `GroupKFold(5, groups=fact)`; AUC is computed
**inside each checkpoint and then averaged**. The forgetting base rate runs 13 %–48 %
across checkpoints, so a feature that merely encodes training time separates the pooled
classes without separating anything within a checkpoint. The pooled AUC is printed
alongside only to show the size of that correction.

### 5b. The supplement's smaller feature set

The 2×2 panel scores three single columns and their combinations under the same
grouped-CV / within-checkpoint-AUC protocol:

| Name | Expression |
|---|---|
| `_pri` — prior state | $\mathbb{1}[C_{t-1} = 0]$, one bit ("just learned") |
| `_chn` — backward churn | `-bwd`, mid-stack, backward window |
| `_gp` — gold probability | `-gold_prob` at $t$ |

---

## 6. Which figure uses what

| Figure | Features |
|---|---|
| **1a / 1b** | `gold_rank_by_layer` → `GRL` (repaired at boundary 18), `correct`, `excluded` |
| **2a** | `ZA_COS` (both windows), `correct` for the transition panels, `gold_prob` for the tercile control |
| **2a supplement** | `ZA_COS` mid-stack backward (`bwd`), `_was` ($C_{t-1}$), `gold_prob` |
| **2b** | the 4 output scalars + the 10 neuron scalars above |

---

## 7. Loaded but not used

Stated explicitly so a reader does not go looking for the figure that consumes them:

* **`is_top1`, `resolution_depth`** — pivoted into `P` by the `FEATURES` list, never read
  again. `resolution_depth` is largely superseded by the full `GRL` trajectory.
* **`top1_prob`** (and the duplicate `pred_entropy` / `gold_prob` copies beside it) — the
  `NEWP` dict is built and never referenced.
* **`prop` (relation), `exposure` (`num_shared_chunks`)** — loaded from the
  fact-properties CSV; no figure in this notebook stratifies by relation or exposure.
* **`mlp_active_frac_by_layer`, `mlp_l1_by_layer`, `mlp_participation_by_layer`,
  `gold_logit_by_layer`, `resid_norm_*`** — present in the parquets, never read; the
  notebook computes its own richer equivalents from the top-128 dump instead.
* **The hidden-state dump** (`hidden_step{S}_{pos}.npy`, input 4) — described in the
  header table as feeding Fig. 3a, but this notebook contains no Fig. 3 cells.
* **`NS` state summaries** — `neuron_summaries("subject_last")` computes `H`, `PR`, `L1`,
  `GEN` at the subject position too, but only `ZS_COS` (its churn) reaches a figure.

---

## 8. Two conventions worth keeping in mind

* **`gold_rank` is 0-based** — 0 means the gold token *is* top-1. Every `+1` before a log
  in this notebook exists for that reason.
* **The prompt split.** `correct` is scored on the 15-shot prompt; every internal
  quantity is measured on the zero-shot cloze, so the answer position is unconditioned
  and comparable across shot counts. Gold is the zero-shot argmax for only 39.8 % of
  correct cells. These are statements about how a fact is *stored*, not about the decode
  that scored it.
