# Analysis guide — how to read the artifacts

This directory is the **consumption** side of the project: `feature_extraction/`
runs the model and writes artifacts; everything here just loads and analyzes
them (CPU is fine — no model needed). Start with this file, then the notebook
[`notebooks/01_explore_and_load.ipynb`](notebooks/01_explore_and_load.ipynb).

The one exception is [`notebooks/02_prompt_playground.ipynb`](notebooks/02_prompt_playground.ipynb),
which *does* load a checkpoint: a debug console for prompts (string in, model's answer
out) at any step, temperature, and decoding setting. Use it when a `generation` in the
artifacts looks wrong and you want to poke at the prompt by hand. It imports nothing
from this repo — just `torch` + `transformers` — so it also runs on its own anywhere.

All paths below are the defaults from [`config.py`](../config.py) (root
`DATA_ROOT = ./data`, override with `LMENT_DATA_ROOT`). Prefer importing the
constants over hard-coding paths:

```python
import sys; sys.path.insert(0, "..")      # repo root (has config.py + src/)
import config
config.RECALL_DIR      # data/artifacts/recall/
config.INTERNAL_DIR    # data/artifacts/internal/
config.FEATURE_DIR     # data/artifacts/features/
config.CHECKPOINT_STEPS  # [0, 10000, ..., 109672]
```

---

## The four artifacts

| artifact | grain | what it holds | built by |
|---|---|---|---|
| `artifacts/recall/recall_step{N}.parquet` | one row per **(fact, checkpoint)** | recall (`correct`, `gold_prob`) **+** internal scalars (`logit_gap_top`, `is_top1`, `resolution_depth`, `mlp_active_frac_mean`, …) **+** per-layer lists (`*_by_layer`) | `eval_checkpoints.py` |
| `artifacts/internal/hidden_step{N}_{pos}.npy` | one **[L+1, H]** stack per fact | full per-layer hidden states, `float16`, `[n_facts, n_layers+1, hidden]`; `pos ∈ {answer, subject_last}` | `eval_checkpoints.py` (unless `--no-hidden`) |
| `artifacts/internal/neurons_step{N}_{pos}.npy` | one **[n_layers, k]** stack per fact | indices of the top-k active MLP neurons per layer, `uint16`; `65535` = position not located | `eval_checkpoints.py` (only with `--top-k`) |
| `artifacts/features/panel.parquet` | one row per **(fact, checkpoint)** | timeline + competition + internal scalars, all as-of that step | `build_feature_table.py` |
| `artifacts/features/facts.parquet` | one row per **fact** | static content + learning-dynamics summary + `atlearn_*` internal snapshot + the `retained` **target** | `build_feature_table.py` |

**Join keys.** Everything keys on `id` (the PopQA fact id). Time-varying tables
also carry `step` (one of `config.CHECKPOINT_STEPS`). The hidden dumps are
row-aligned to a companion `hidden_step{N}_ids.npy`, so you map `id -> row`
yourself (see below).

---

## Loading the tables

```python
import pandas as pd, numpy as np, config

# One checkpoint
df = pd.read_parquet(config.RECALL_DIR / "recall_step10000.parquet")

# The whole sweep as a long panel (fact × checkpoint)
panel = pd.concat(
    [pd.read_parquet(config.RECALL_DIR / f"recall_step{s}.parquet")
     for s in config.CHECKPOINT_STEPS
     if (config.RECALL_DIR / f"recall_step{s}.parquet").exists()],
    ignore_index=True,
)

# The modeling table (one row per fact)
facts = pd.read_parquet(config.FEATURE_DIR / "facts.parquet")
```

A fact's **trajectory** is just the panel filtered and sorted by step:

```python
traj = panel[panel.id == fact_id].sort_values("step")
traj[["step", "correct", "gold_prob", "logit_gap_top", "resolution_depth"]]
```

---

## Loading the hidden states

The dump is a plain `float16` array of shape `[n_facts, n_layers+1, hidden]` per
position, memmap-friendly (don't read the whole thing into RAM):

```python
import numpy as np, config, json

step, pos = 10000, "answer"
H   = np.load(config.INTERNAL_DIR / f"hidden_step{step}_{pos}.npy", mmap_mode="r")  # [N, L+1, H] f16
ids = np.load(config.INTERNAL_DIR / f"hidden_step{step}_ids.npy")
meta = json.loads((config.INTERNAL_DIR / f"hidden_step{step}_meta.json").read_text())

row_of = {int(fid): i for i, fid in enumerate(ids)}   # fact id -> row index
vec = np.asarray(H[row_of[fact_id]], dtype=np.float32) # [L+1, hidden] for that (ckpt, fact)
```

Notes:
- **`pos="answer"`** is the cloze's last token (predicts the object) — the state
  where the answer is formed. **`pos="subject_last"`** is the subject's final
  token — use it for probing and drift.
- Rows for facts whose position couldn't be located are **all-NaN**. Drop them:
  `valid = ~np.isnan(H[:, 0, 0])`.
- Cast to `float32` before doing math — `float16` overflows/underflows easily in
  dot products and norms.
- Layer axis is length `n_layers+1`: index `0` is the embedding output, index `k`
  is the residual stream after block `k`.

To stack the **same fact across checkpoints** (for drift / trajectories in
representation space):

```python
def fact_across_steps(fact_id, pos="subject_last"):
    out = {}
    for s in config.CHECKPOINT_STEPS:
        p = config.INTERNAL_DIR / f"hidden_step{s}_{pos}.npy"
        if not p.exists():
            continue
        ids = np.load(config.INTERNAL_DIR / f"hidden_step{s}_ids.npy")
        r = {int(f): i for i, f in enumerate(ids)}
        if fact_id in r:
            out[s] = np.asarray(np.load(p, mmap_mode="r")[r[fact_id]], np.float32)
    return out   # {step: [L+1, hidden]}
```

---

## Loading the top-k neuron dump

Written only when the sweep ran with `--top-k K`. Per position, a `uint16` array
of shape `[n_facts, n_layers, k]` holding the **indices** of the k most active
feed-forward neurons per layer (descending `|activation|`). It exists to answer
*which* neurons a fact uses — the circuit — not just how many (that's the
`mlp_active_frac_*` / `mlp_participation_*` scalars in the parquet).

```python
import numpy as np, config, json

step, pos = 10000, "answer"
N   = np.load(config.INTERNAL_DIR / f"neurons_step{step}_{pos}.npy", mmap_mode="r")  # [F, L, k] uint16
ids = np.load(config.INTERNAL_DIR / f"neurons_step{step}_ids.npy")   # one ids file per step
meta = json.loads((config.INTERNAL_DIR / f"neurons_step{step}_meta.json").read_text())
MISSING = meta["missing_sentinel"]      # 65535 -> position wasn't located for that fact

row_of = {int(f): i for i, f in enumerate(ids)}
top = np.asarray(N[row_of[fact_id]])    # [n_layers, k] neuron indices for this (ckpt, fact)
```

- A row is **all-`MISSING`** if the position couldn't be located (e.g. subject
  not found). Drop with `valid = (N[:, 0, 0] != MISSING)`.
- Order is by activation magnitude, so `top[layer, 0]` is the single most active
  neuron in that layer — but for set-overlap questions treat each row as a **set**
  (order doesn't matter).

---

## Analysis recipes (concepts)

These are the general moves; adapt column/step choices to your question.

### 1. Acquisition & forgetting curves
The backbone of the project. Average `correct` (or `gold_prob`) over facts at
each step, or per relation / popularity bucket, to see *when* knowledge appears
and whether it decays.

```python
curve = panel.groupby("step")[["correct", "gold_prob"]].mean()
```
A fact is **retained** if it's ever correct and still correct at the final step,
**forgotten** if it flips back to wrong — that's the `retained` target in
`facts.parquet` (see `src/features/learning.py::retention_target`).

### 2. "Distance from the top" over training (`logit_gap_top`)
`logit_gap_top = max_logit − gold_logit` at the answer position: how far the
correct token is below the model's top pick. `0` ⇔ the fact is top-1. Watching it
fall toward 0 (and `resolution_depth` shrink) shows a fact being *acquired*;
watching it rise again shows it being *overwritten*.

```python
traj.plot(x="step", y="logit_gap_top")   # per fact; 0 == answered correctly (greedy)
```
> ⚠️ `logit_gap_top`, `gold_rank`, `is_top1`, `answer_margin` are **first-token**
> quantities. For multi-token answers filter `gold_ntokens == 1` (or use the
> length-normalized `gold_prob` for the whole answer).

### 3. Depth of knowledge (logit lens)
`resolution_depth` / `resolution_depth_frac` = the first layer where the gold
token becomes top-1 under the logit lens; `gold_rank_by_layer` is the full
per-layer trajectory. Facts resolved in **earlier** layers are typically stored
more robustly. You can replay the lens yourself from the hidden dump if you kept
it (needs the model's final norm + LM head — see `src/features/internal.py`).

### 4. Representation drift between checkpoints
How much a fact's `subject_last` state *moves* from one checkpoint to the next —
large movement after first-learned suggests overwriting.

```python
from src.features import internal
states = fact_across_steps(fact_id, pos="subject_last")   # {step: [L+1, H]}
steps = sorted(states)
for a, b in zip(steps, steps[1:]):
    print(a, "->", b, internal.representation_drift(states[a], states[b]))
```

### 5. Linear probes (is the answer readable from the subject?)
Collect the `subject_last` states at one checkpoint into `[n_facts, L+1, H]`,
label each fact by its object, and fit a per-layer linear classifier — a high
peak layer means the fact is *linearly* stored.

```python
from src.features import internal
H = np.load(config.INTERNAL_DIR / "hidden_step109672_subject_last.npy")  # [N, L+1, H]
ids = np.load(config.INTERNAL_DIR / "hidden_step109672_ids.npy")   # one ids file per step
# align labels to ids, drop NaN rows (X_by_layer = H[valid]), then:
# accs = internal.train_layer_probes(X_by_layer, y)   # cross-val accuracy per layer
```

### 6. Neuron / capacity usage — *how much* machinery
Three scalar/per-layer signals in the parquet, at the answer position (and the
`_subj` variants at the subject token):
- `mlp_active_frac_*` — fraction of neurons with `|activation| > NEURON_EPS`
  (a hard-threshold count).
- `mlp_participation_*` — the **participation ratio** `(Σa²)² / Σa⁴`, i.e. the
  *effective number* of neurons carrying the activation energy. Scale-free: near
  `1` means a few neurons dominate (concentrated / specialized storage), high
  means the fact is spread across many neurons. Often more informative than the
  raw fraction.
- `resid_norm_by_layer` — how much the residual stream has grown by each layer.

Compare retained vs. forgotten facts, or watch the profile shift over training
(e.g. does a fact's storage *concentrate* — participation ratio dropping — as it's
learned?).

### 6b. Neuron *identity* — *which* machinery (needs `--top-k`)
The scalars above can't tell whether a fact keeps using the **same** neurons. The
`neurons_step{N}_{pos}.npy` dump lets you ask that directly. Treat each layer's
top-k as a set and measure overlap.

```python
def jaccard(a, b):                       # a, b: 1-D arrays of neuron indices
    A, B = set(a.tolist()), set(b.tolist())
    return len(A & B) / len(A | B) if (A or B) else np.nan

def load_top(step, pos, fact_id):
    N = np.load(config.INTERNAL_DIR / f"neurons_step{step}_{pos}.npy", mmap_mode="r")
    ids = np.load(config.INTERNAL_DIR / f"neurons_step{step}_ids.npy")
    return np.asarray(N[{int(f): i for i, f in enumerate(ids)}[fact_id]])  # [L, k]

# Circuit CHURN: how stable is a fact's neuron set across consecutive checkpoints?
steps = config.CHECKPOINT_STEPS
for a, b in zip(steps, steps[1:]):
    ta, tb = load_top(a, "answer", fact_id), load_top(b, "answer", fact_id)
    per_layer = [jaccard(ta[l], tb[l]) for l in range(ta.shape[0])]
    print(a, "->", b, "mean layer overlap:", np.nanmean(per_layer))
```
Ideas: **churn** (low across-checkpoint overlap after first-learned ⇒ the fact is
being rewritten — candidate forgetting signal); **sharing** (cross-*fact* overlap
⇒ facts stored in shared vs. dedicated neurons); **answer-vs-subject** overlap
(does the object get written into the same neurons that encode the subject?).

### 7. Modeling retention (Step C) — mind the leakage
Train on `facts.parquet`. **Drop every `leaky_*` column** (they encode the
target). The `atlearn_*` columns are the internal state **at first-learned**, so
they're safe predictors; do **not** swap in final-checkpoint internals — those
leak `retained` the same way `leaky_*` does.

```python
facts = pd.read_parquet(config.FEATURE_DIR / "facts.parquet")
X = facts.drop(columns=[c for c in facts if c.startswith("leaky_")]
                       + ["retained", "id", "subj", "obj"])
y = facts["retained"]           # 1 retained / 0 forgotten / NaN never-learned
mask = y.notna()                # keep only ever-learned facts for the binary task
```

---

## Gotchas

- **NaN is meaningful**, not missing-at-random: a temporal feature is NaN before
  the fact first appears; a learning feature is NaN if the fact was never
  correct; a hidden-state row is NaN if the token wasn't locatable
  (`subject_found == False`). Filter deliberately.
- **`excluded` facts** (answer already in the query) should be dropped from most
  analyses — olmes never scores them.
- **f16 precision**: the dumps are `float16` to save disk; upcast to `float32`
  for any arithmetic.
- **Which positions exist** depends on how the sweep was run (`--positions`,
  `--no-hidden`). Check `hidden_step{N}_meta.json`.
- **Per-layer list columns** (`*_by_layer`) live only in the recall parquet, not
  in `panel`/`facts`; explode them with `np.vstack(df["gold_rank_by_layer"])`.
