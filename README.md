# Predicting fact retention vs. forgetting in LMEnt

Holding fact *frequency* fixed, which properties of the corpus, the training
schedule, and the model's internal representation predict whether a fact is
**retained** or **forgotten** during pretraining?

Built on the [LMEnt](https://github.com/dhgottesman/LMEnt) suite
([paper](https://arxiv.org/abs/2509.03405)) — the `LMEnt-1B-1E` model with its
12 intermediate checkpoints, the entity-annotated corpus, and the
[`popqa-kas`](https://huggingface.co/datasets/dhgottesman/popqa-kas) dataset.

## Layout

```
config.py                  # all IDs, checkpoint steps, paths, LR schedule (VERIFY tags)
src/
  data.py                  # load dataset + checkpoints + tokenizer, prompt templates
  schedule.py              # batch_indices -> chunk->steps index; LR at step
  recall.py                # per-checkpoint substring recall + gold-answer prob
  features/
    timeline.py            # recency, spacing/burstiness, LR-at-appearance, epoch position
    competition.py         # subject/answer/shared counts; interference post-learning
    text.py                # answer length, relation, mention explicitness, phrasing variety
    learning.py            # first-learned step, confidence, learning speed (+ leaky_* target-likes)
    internal.py            # logit-lens depth, answer margin, hidden-state probe, drift, paraphrase
scripts/
  eval_checkpoints.py      # Step A.1: recall sweep over all checkpoints  -> artifacts/recall/
  build_feature_table.py   # Step A.2/B: assemble panel.parquet + facts.parquet
notebooks/
  01_explore_and_load.ipynb  # end-to-end demo on a single fact
```

## Feature groups (mapped to the proposal)

| Group | Module | Notes |
|-------|--------|-------|
| When the fact showed up | `features/timeline.py` | `steps_since_last` (recency) is likely the strongest; plus spacing/burstiness, LR at appearance, epoch position |
| Competition from other facts | `features/competition.py` | subject & answer counts **separately**; interference *after* first-learned |
| What the text looks like | `features/text.py` | answer token length, explicit (EL) vs. coref mentions, phrasing variety, subj–answer distance |
| How the fact was learned | `features/learning.py` | first-learned step, confidence there, learning speed. `leaky_*` = forgetting/oscillation counts — **EDA only, drop before modeling** |
| Inside the model | `features/internal.py` | logit-lens resolution depth, answer margin, linear probe, representation drift, paraphrase stability |

## Setup

```bash
pip install -r requirements.txt
export LMENT_DATA_ROOT=/path/to/storage      # checkpoints, corpus, artifacts
```

Download from HuggingFace: the model `dhgottesman/LMEnt-1B-1E` (checkpoints are
loaded on demand by subfolder) and the `LMEnt-Dataset` pieces you need —
`dataset-cache` (batch indices, ~0.6 GB) at minimum; `dataset-metadata`
(mentions) only for the text-mention features.

## Run order

```bash
# Step A.1 — recall + gold_prob at every checkpoint (GPU; one per checkpoint)
python scripts/eval_checkpoints.py            # or --steps 0 10000 ... as a Slurm array

# Step A.2 / B — assemble feature tables
python scripts/build_feature_table.py         # -> artifacts/features/{panel,facts}.parquet

# Step C — fit interpretable model on facts.parquet (drop leaky_* columns first)
```

The prediction target: a fact that is **ever** correct and still correct at the
final checkpoint = *retained* (1); ever correct but wrong at the end =
*forgotten* (0); never correct = excluded. See `learning.retention_target`.

## Feature dictionary (output columns)

`NaN` appears wherever a feature is undefined for a fact at a given checkpoint
(e.g. temporal features before the fact has appeared, or learning features for a
fact that is never correct).

### `recall_step{N}.parquet` — one row per fact, per checkpoint (`src/recall.py`)

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `id` | int64 | dataset fact id | joins to `popqa-kas` |
| `step` | int | one of `CHECKPOINT_STEPS` (0 … 109672) | checkpoint evaluated |
| `correct` | bool | `True`/`False` | greedy decode contains an acceptable answer |
| `gold_prob` | float | `[0, 1]` | length-normalized prob of the gold object tokens |

### `panel.parquet` — one row per (fact × checkpoint) (`scripts/build_feature_table.py`)

Contains `id`, `step`, `correct`, `gold_prob` (above) **+** all *timeline* and
*competition* columns below, recomputed relative to that checkpoint's `step`.

**Timeline** (`features/timeline.py`) — all relative to the checkpoint step:

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `n_appearances` | int | `≥ 0` | cumulative shared-chunk appearances seen so far |
| `steps_since_last` | int / NaN | `≥ 0` | steps since the fact was last seen (**recency — likely strongest**) |
| `first_seen_step` | int / NaN | `0 … 109672` | step of first appearance |
| `last_seen_step` | int / NaN | `0 … 109672` | step of most recent appearance ≤ checkpoint |
| `span` | int / NaN | `≥ 0` | `last_seen − first_seen` (spread of appearances) |
| `gap_mean` | float / NaN | `≥ 0` | mean steps between consecutive appearances |
| `gap_std` | float / NaN | `≥ 0` | std of those gaps |
| `gap_cv` | float / NaN | `≥ 0` | coefficient of variation (`std/mean`) |
| `max_gap` | float / NaN | `≥ 0` | largest gap between appearances |
| `burstiness` | float / NaN | `[-1, 1]` | −1 = evenly spaced, 0 = random, +1 = bursty |
| `lr_first` | float / NaN | `[0, LR_PEAK]` | learning rate at first appearance |
| `lr_last` | float / NaN | `[0, LR_PEAK]` | learning rate at last appearance |
| `lr_mean` | float / NaN | `[0, LR_PEAK]` | mean LR across appearances |
| `first_epoch_frac` | float / NaN | `[0, 1]` | first appearance as fraction of the epoch (early/mid/late) |
| `last_epoch_frac` | float / NaN | `[0, 1]` | last appearance as fraction of the epoch |

**Competition / interference** (`features/competition.py`):

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `subject_num_chunks` | int | `≥ 0` | chunks mentioning the subject |
| `answer_num_chunks` | int | `≥ 0` | chunks mentioning the answer |
| `num_shared_chunks` | int | `≥ 0` | chunks with both (where the fact can be learned) |
| `subject_only_chunks` | int | `≥ 0` | subject chunks that do **not** state the fact |
| `answer_only_chunks` | int | `≥ 0` | answer chunks that do **not** state the fact |
| `s_pop` | int | `≥ 0` | subject Wikipedia popularity |
| `o_pop` | int | `≥ 0` | object Wikipedia popularity |
| `subject_interference_seen` | int | `≥ 0` | subject-only appearances seen by this checkpoint |
| `answer_interference_seen` | int | `≥ 0` | answer-only appearances seen by this checkpoint |
| `subject_interference_post_learn` | int / NaN | `≥ 0` | subject-only appearances **after** first-learned (overwriting pressure) |
| `answer_interference_post_learn` | int / NaN | `≥ 0` | answer-only appearances after first-learned |

### `facts.parquet` — one row per fact; Step C trains on this

Static identifiers + content features (evaluated at the **final** checkpoint) +
learning-dynamics summary + target. Includes all *competition* and *timeline*
columns above (at `FINAL_STEP`) plus:

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `id` / `subj` / `obj` / `prop` | int / str | — | identifiers |
| `retained` | int / NaN | `1` retained, `0` forgotten, `NaN` never-learned (excluded) | **prediction target** |

**Text content** (`features/text.py::cheap_text_features`):

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `relation` | str | `occupation`, `place of birth`, `country`, `capital of`, `author`, `director`, `composer`, `sport`, `religion`, `color`, `genre`, `father`, `mother`, `producer`, `screenwriter`, `capital` | PopQA relation type |
| `answer_token_len` | int | `≥ 1` | gold answer length in tokens (harder to say = more fragile) |
| `answer_char_len` | int | `≥ 1` | answer length in characters |
| `answer_word_len` | int | `≥ 1` | answer length in words |

**Learning dynamics** (`features/learning.py`):

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `first_learned_step` | int / NaN | `0 … 109672` | first checkpoint where correct |
| `confidence_at_first_known` | float / NaN | `[0, 1]` | gold prob at first-learned (**likely strong**) |
| `steps_to_learn` | int / NaN | `≥ 0` | steps from first exposure to first correct |
| `max_gold_prob` | float / NaN | `[0, 1]` | peak gold prob ever reached |
| `leaky_forgetting_events` | int | `≥ 0` | ⚠️ 1→0 flips — **target leakage, EDA only** |
| `leaky_relearning_events` | int | `≥ 0` | ⚠️ 0→1 flips — **EDA only** |
| `leaky_oscillation_count` | int | `≥ 0` | ⚠️ total flips — **EDA only** |
| `leaky_ever_correct` | bool | — | ⚠️ correct at any checkpoint — **EDA only** |
| `leaky_correct_final` | bool | — | ⚠️ correct at the final checkpoint (= target) — **EDA only** |

> Drop every `leaky_*` column before fitting the Step C model — they encode the
> answer you're predicting.

### Mention & internal features (computed on demand, not in the parquets by default)

**Mention-based text** (`features/text.py::mention_features`, needs `dataset-metadata`):

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `frac_explicit_subject` | float / NaN | `[0, 1]` | fraction of subject mentions that are entity-links (explicit) |
| `frac_coref_subject` | float / NaN | `[0, 1]` | fraction that are coreference (indirect, weaker signal) |
| `n_subject_mentions` | int | `≥ 0` | subject mentions across shared chunks |
| `subj_answer_distance` | float / NaN | `≥ 0` | mean token distance between subject and answer in-chunk |
| `phrasing_variety` | float / NaN | `(0, 1]` | distinct chunk texts / total (1 = all different) |

**Inside the model** (`features/internal.py`, per fact per checkpoint, GPU):

| function → column | type | range / values | meaning |
|-------------------|------|----------------|---------|
| `logit_lens_features` → `resolution_depth` | int / NaN | `0 … n_layers` | first layer where gold answer is top-1 |
| `logit_lens_features` → `resolution_depth_frac` | float / NaN | `[0, 1]` | depth normalized by layer count |
| `logit_lens_features` → `answer_margin` | float / NaN | ℝ (logit units) | gold logit − runner-up logit at final layer |
| `representation_drift` → `drift_cos_mean` | float / NaN | `[-1, 1]` | mean per-layer cosine sim of subject token between checkpoints |
| `representation_drift` → `drift_cos_last` | float / NaN | `[-1, 1]` | same at the final layer |
| `paraphrase_stability` → `paraphrase_n` | int | `1 … 3` | number of phrasings tried |
| `paraphrase_stability` → `paraphrase_prob_mean` | float / NaN | `[0, 1]` | mean gold prob across phrasings |
| `paraphrase_stability` → `paraphrase_prob_std` | float / NaN | `≥ 0` | spread across phrasings (high = phrasing-sensitive) |
| `paraphrase_stability` → `paraphrase_correct_frac` | float | `[0, 1]` | fraction of phrasings answered correctly |
| `paraphrase_stability` → `paraphrase_all_correct` | bool | — | correct under every phrasing |
| `train_layer_probes` → (array) | float[] | each `[0, 1]` | per-layer linear-probe accuracy for the answer from the subject hidden state |

## ⚠️ Verify against the real download

Everything below was inferred from the repo/dataset pages and must be confirmed
once the data is local (all isolated in `config.py` or tagged `# VERIFY`):

- **LR schedule** constants (`LR_PEAK`, `LR_WARMUP_STEPS`, family) — read from the
  training config; only the *shape* feeds the feature, but get the family right.
- **`batch_indices` path & shape** — assumed `dataset-cache/batch_indices.npy`,
  step-indexed array of chunk-ID arrays.
- **PopQA-KAS column names** — `subject_chunks`/`shared_chunks`/etc.; check
  `dataset.features`.
- **Mention schema** (`features/text.py::MentionProvider`) — ReFinED (explicit)
  vs. Maverick (coref); token vs. char offsets.
- **Model internals** (`features/internal.py::_final_norm_and_head`) — final norm
  and `lm_head` attribute names for the OLMo-core architecture.
- **Relation templates** (`data.py::RELATION_TEMPLATES`) — confirm the exact
  `prop` strings used in the data.
