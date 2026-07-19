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
