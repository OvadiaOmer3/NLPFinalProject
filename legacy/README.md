# legacy/ — the original "panel + facts" feature pipeline (superseded)

**Status: kept for reference; none of the submitted results depend on it.**

This is the proposal's first design: build one table of per-fact features
(timeline, competition, text, learning dynamics) and fit an interpretable model
of *retained vs. forgotten* on it. The four sub-projects in the repo root
(`frequency_forgetting/`, `neuron_forgetting/`, `token_strain/`,
`explicit_mentions/`) each re-derived what they needed directly from the recall
parquets instead, so nothing imports these modules and nothing reads their
outputs (`panel.parquet`, `facts.parquet`). The only live consumer of anything
here was the mention-schema test, which moved along with it.

| file | purpose |
|---|---|
| `build_feature_table.py` | Step A.2/B: assemble `panel.parquet` (fact × checkpoint) and `facts.parquet` (one row per fact) from `data/artifacts/recall/recall_step*.parquet` |
| `schedule.py` | `batch_indices` → chunk→steps inverted index (`config.CHUNK_INDEX_PATH`); LR at a step from `config.LR_*` |
| `features/timeline.py` | recency, spacing/burstiness, LR-at-appearance, epoch position |
| `features/competition.py` | subject/answer/shared chunk counts; interference after first-learned |
| `features/text.py` | answer length, relation, mention explicitness (needs LMEnt's tokenized corpus + `olmo_core`), phrasing variety |
| `features/learning.py` | first-learned step, confidence there, learning speed; `leaky_*` target-likes (EDA only) |
| `test_mention_features.py` | `text.mention_features` against LMEnt's real mention schema |

The `config.py` constants that exist only for this code: `LR_PEAK`,
`LR_WARMUP_STEPS`, `LR_ALPHA_F`, `LR_MIN`, `BATCH_INDICES_PATH`, `TOKENIZED_GLOB`,
`CHUNK_INDEX_PATH`.

## Running it

```bash
# needs data/artifacts/recall/recall_step*.parquet from feature_extraction/eval_checkpoints.py
# (a full run, not --no-internal) and dataset-cache/batch_indices_epoch_1.npy
python legacy/build_feature_table.py            # -> data/artifacts/features/{panel,facts}.parquet
python -m pytest legacy/test_mention_features.py -q
```

The prediction target: a fact that is **ever** correct and still correct at the
final checkpoint = *retained* (1); ever correct but wrong at the end =
*forgotten* (0); never correct = excluded. See `learning.retention_target`.

## Feature groups (mapped to the proposal)

| Group | Module | Notes |
|-------|--------|-------|
| When the fact showed up | `features/timeline.py` | `steps_since_last` (recency); plus spacing/burstiness, LR at appearance, epoch position |
| Competition from other facts | `features/competition.py` | subject & answer counts **separately**; interference *after* first-learned |
| What the text looks like | `features/text.py` | answer token length, explicit (EL) vs. coref mentions, phrasing variety, subj–answer distance |
| How the fact was learned | `features/learning.py` | first-learned step, confidence there, learning speed. `leaky_*` = forgetting/oscillation counts — **EDA only, drop before modeling** |

## Feature dictionary

`NaN` appears wherever a feature is undefined for a fact at a given checkpoint
(e.g. temporal features before the fact has appeared, or learning features for a
fact that is never correct).

### `panel.parquet` — one row per (fact × checkpoint)

Contains `id`, `step`, `correct`, `gold_prob`, the **internal-state scalar
columns** at that step (`is_top1`, `gold_rank`, `logit_gap_top`, `answer_margin`,
`pred_entropy`, `resolution_depth`, `mlp_active_frac_mean`, … whichever were
produced by the sweep; see the root README's `recall_step{N}.parquet` dictionary)
**+** all *timeline* and *competition* columns below, recomputed relative to that
checkpoint's `step`.

**Timeline** (`features/timeline.py`) — all relative to the checkpoint step:

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `n_appearances` | int | `≥ 0` | cumulative shared-chunk appearances seen so far |
| `steps_since_last` | int / NaN | `≥ 0` | steps since the fact was last seen (recency) |
| `first_seen_step` | int / NaN | `0 … 658032` | step of first appearance |
| `last_seen_step` | int / NaN | `0 … 658032` | step of most recent appearance ≤ checkpoint |
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

### `facts.parquet` — one row per fact; the modeling table

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
| `answer_token_len` | int | `≥ 1` | gold answer length in tokens |
| `answer_char_len` | int | `≥ 1` | answer length in characters |
| `answer_word_len` | int | `≥ 1` | answer length in words |

**Learning dynamics** (`features/learning.py`):

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `first_learned_step` | int / NaN | `0 … 658032` | first checkpoint where correct |
| `confidence_at_first_known` | float / NaN | `[0, 1]` | gold prob at first-learned |
| `steps_to_learn` | int / NaN | `≥ 0` | steps from first exposure to first correct |
| `max_gold_prob` | float / NaN | `[0, 1]` | peak gold prob ever reached |
| `leaky_forgetting_events` | int | `≥ 0` | ⚠️ 1→0 flips — **target leakage, EDA only** |
| `leaky_relearning_events` | int | `≥ 0` | ⚠️ 0→1 flips — **EDA only** |
| `leaky_oscillation_count` | int | `≥ 0` | ⚠️ total flips — **EDA only** |
| `leaky_ever_correct` | bool | — | ⚠️ correct at any checkpoint — **EDA only** |
| `leaky_correct_final` | bool | — | ⚠️ correct at the final checkpoint (= target) — **EDA only** |

> Drop every `leaky_*` column before fitting a model — they encode the answer
> you're predicting.

**Internal state at first-learned** (`atlearn_*`, from `internal.extract_fact`) —
the internal-state scalars snapshotted at the fact's `first_learned_step`, i.e.
*how the fact looked in the model the moment it was first acquired*
(`atlearn_logit_gap_top`, `atlearn_resolution_depth`, `atlearn_pred_entropy`,
`atlearn_mlp_active_frac_mean`, …). `NaN` for never-learned facts. These are
**non-leaky** — measured before the end of training.

> ⚠️ Do **not** substitute the *final*-checkpoint internal state here: because
> `retained` is defined by final-checkpoint correctness, final-step internals
> would leak the target, like `leaky_*`. `build_feature_table.py` deliberately
> reads the first-learned step instead.

**Mention-based text** (`features/text.py::mention_features`) — also in
`facts.parquet`. These need LMEnt's tokenized corpus + `olmo_core` to read each
chunk's aggregated ReFinED + Maverick mentions; when the corpus isn't present
(PopQA-only setup) the columns are still emitted, as `NaN`. Entities are matched
by **Wikidata QID** (`s_uri` / `o_uri`, e.g. `"Q42"` — *not* `subj_id`/`obj_id`).

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `frac_explicit_subject` | float / NaN | `[0, 1]` | fraction of subject mentions that are hyperlink / entity-link evidence (explicit) |
| `frac_coref_subject` | float / NaN | `[0, 1]` | fraction that are coreference / pronoun (indirect, weaker signal) |
| `n_subject_mentions` | int | `≥ 0` | subject mentions across shared chunks (score ≥ `MIN_CANDIDATE_SCORE`) |
| `subj_answer_distance` | float / NaN | `≥ 0` | mean **character** distance (mention midpoints) between subject & answer in-chunk |
| `phrasing_variety` | float / NaN | `(0, 1]` | distinct chunk texts / total (1 = all different) |

> Explicit vs. coref comes from each matched candidate's `scores_by_source`
> (`hyperlinks`/`entity_linking` = explicit; `coref`/`coref_cluster` = indirect);
> a pronoun surface form is always counted as coref. Provider access follows
> LMEnt's `retrieval-index/create_es_index.py` (`OlmoCoreMentionProvider`);
> `DictMentionProvider` backs the tests in `test_mention_features.py`.

**Mention schema** (`features/text.py`) — read from LMEnt's own reader
(`retrieval-index/create_es_index.py`): per-chunk `metadata["entities"]`, each
mention having **char** offsets, `text_mention`, and `candidates[{qid,
aggregated_score, scores_by_source{hyperlinks, entity_linking, coref,
coref_cluster}}]`. Entities are Wikidata QIDs (`s_uri`/`o_uri`). `mention_features`
is implemented + unit-tested against this schema; it only *runs* when the 47 GB
tokenized corpus + `olmo_core` are present (else NaN columns).
