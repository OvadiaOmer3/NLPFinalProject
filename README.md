# Predicting fact retention vs. forgetting in LMEnt

Holding fact *frequency* fixed, which properties of the corpus, the training
schedule, and the model's internal representation predict whether a fact is
**retained** or **forgotten** during pretraining?

Built on the [LMEnt](https://github.com/dhgottesman/LMEnt) suite
([paper](https://arxiv.org/abs/2509.03405)) — the `LMEnt-1B-1E` model with its
12 intermediate checkpoints, the entity-annotated corpus, and the
[`popqa-kas`](https://huggingface.co/datasets/dhgottesman/popqa-kas) dataset.

## Layout

The repo is organized around two phases: **feature_extraction/** runs the model
to *produce* artifacts, **analysis/** *consumes* them. `src/` is the shared
library both phases import.

```
config.py                  # all IDs, checkpoint steps, paths, LR schedule, few-shot settings
src/                       # shared library (imported by both phases)
  lment_popqa.py           # SINGLE home for the olmes coupling: cloze prompts (by prop_id),
                           #   few-shot context, exact-match scoring, generation kwargs
  data.py                  # load dataset + checkpoints + tokenizer; prompt wrappers -> lment_popqa
  schedule.py              # batch_indices -> chunk->steps index; LR at step (real kas_config)
  recall.py                # per-checkpoint cloze recall (olmes scoring) + gold-answer prob
  features/
    timeline.py            # recency, spacing/burstiness, LR-at-appearance, epoch position
    competition.py         # subject/answer/shared counts; interference post-learning
    text.py                # answer length, relation, mention explicitness, phrasing variety
    learning.py            # first-learned step, confidence, learning speed (+ leaky_* target-likes)
    internal.py            # one-pass extractor: logit-lens depth, next-token geometry,
                           #   MLP activity, hidden-state probe/drift/paraphrase (+ f16 dump)
feature_extraction/        # PRODUCES artifacts (run on GPU)
  eval_checkpoints.py      # Step A.1: recall + internal-state sweep  -> artifacts/recall/ (+ internal/)
  build_feature_table.py   # Step A.2/B: assemble panel.parquet + facts.parquet
analysis/                  # CONSUMES artifacts (CPU / notebooks)
  README.md                # how to load & analyze the artifacts (start here)
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
| Inside the model | `features/internal.py` | **one forward pass per (fact, checkpoint)**: logit-lens depth, next-token geometry (margin, top-logit gap, rank, entropy), per-layer residual/MLP-neuron activity (active fraction + participation ratio, at answer & subject), plus f16 hidden-state and optional top-k neuron-index dumps for offline probe / drift / circuit-overlap |

## Setup

```bash
pip install -r requirements.txt
export LMENT_DATA_ROOT=/path/to/storage      # checkpoints, corpus, artifacts
```

Download from HuggingFace: the model `dhgottesman/LMEnt-1B-1E` (checkpoints are
loaded on demand by subfolder) and the `LMEnt-Dataset` pieces you need —
`dataset-cache` (batch indices, ~0.6 GB) at minimum; `dataset-metadata`
(mentions) only for the text-mention features.

**Prompts & scoring come from LMEnt's olmes fork.** `src/lment_popqa.py` uses the
real `PopQACloze` task when olmes is installed (authoritative), and otherwise a
faithful fallback transcribed from the fork so it runs without it. To use the
real harness:

```bash
cd LMEnt
git submodule update --init olmes            # prompts + scoring (PopQACloze)
git submodule update --init OLMo-core         # only to read kas_config.json / scheduler
pip install -e olmes                          # install AFTER the env so the fork wins
```

## Run order

```bash
# Step A.1 — recall + internal states at every checkpoint (GPU; one per checkpoint)
python feature_extraction/eval_checkpoints.py            # or --steps 0 10000 ... as a Slurm array
#   --num-shots N          in-context demos (default 15; 0 = bare cloze). Demos come from a
#                          held-out pool of one fact per relation, never the eval set, and a
#                          fact is never shown its OWN relation -- hence 15 of 16. The pool's
#                          16 subjects are dropped at EVERY shot count, so 0-shot and n-shot
#                          score an identical fact set. Output is suffixed `_{N}shot` so runs
#                          at different N don't overwrite each other.
#   --fewshot-keep-same-prop    one fixed block incl. the query's own relation (upstream)
#   --fewshot-source upstream   use LMEnt's verbatim demos (leaky; parity runs only)
#   --no-hidden            skip the f16 hidden-state dump (scalar features only)
#   --positions answer     dump fewer positions (default: answer subject_last)
#   --top-k 64             also dump top-k active-neuron indices (off by default)
#   --limit 100            smoke test on the first 100 facts
#   --output-dir D:/out    write to an explicit root instead of $LMENT_DATA_ROOT/artifacts
#                          (cwd-independent -- avoids output landing in the wrong
#                          place if you run the script from outside the repo root)
#   --min-shared-chunks 1  drop facts whose subject & answer never co-occur in a
#                          shared_chunk (~17% of PopQA-KAS, 2,450/14,267) -- they can
#                          never get a first_learned_step or timeline features since
#                          there's no exposure to track. Default 0 = keep everything.
# -> artifacts/recall/recall_step{N}.parquet   (recall + internal scalar/per-layer features)
# -> artifacts/internal/hidden_step{N}_*.npy   (f16 [n_facts, n_layers+1, hidden] per position)
# -> artifacts/internal/neurons_step{N}_*.npy  (uint16 [n_facts, n_layers, k]; only with --top-k)

# Step A.2 / B — assemble feature tables
python feature_extraction/build_feature_table.py         # -> artifacts/features/{panel,facts}.parquet

# Step C — fit interpretable model on facts.parquet (drop leaky_* columns first)
```

**One inference per (fact, checkpoint).** `eval_checkpoints.py` touches the model
exactly twice per fact: once for the greedy decode `correct` needs, and once for a
teacher-forced forward that yields *everything else* — `gold_prob`, logit-lens
depth, next-token geometry, per-layer residual/MLP activity, and the hidden-state
dump. (A true single pass is possible only if you redefine `correct` as
"first answer token is top-1", i.e. the `is_top1` column — but that changes the
paper's metric, so the decode is kept.)

The prediction target: a fact that is **ever** correct and still correct at the
final checkpoint = *retained* (1); ever correct but wrong at the end =
*forgotten* (0); never correct = excluded. See `learning.retention_target`.

## Feature dictionary (output columns)

`NaN` appears wherever a feature is undefined for a fact at a given checkpoint
(e.g. temporal features before the fact has appeared, or learning features for a
fact that is never correct).

### `recall_step{N}.parquet` — one row per fact, per checkpoint (`feature_extraction/eval_checkpoints.py`)

Recall columns (from the greedy decode) **+** internal-state columns (from the
single teacher-forced forward in `internal.extract_fact`).

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `id` | int64 | dataset fact id | joins to `popqa-kas` |
| `step` | int | one of `CHECKPOINT_STEPS` (0 … 109672) | checkpoint evaluated |
| `correct` | bool | `True`/`False` | olmes `PopQACloze` exact-match on the cloze generation |
| `excluded` | bool | `True`/`False` | olmes drops these (answer already in the query) — filter out in EDA |
| `gold_prob` | float | `[0, 1]` | **length-normalized** prob over **all** gold object tokens (zero-shot cloze) |
| `prompt` | str | cloze text | exact prompt fed to the model for `correct` (audit/repro) |
| `generation` | str | model output | model's decoded answer, post stop-truncation (audit/repro) |

**Internal-state scalars** (`internal.extract_fact`) — read at the *answer
position* (the cloze's last token, which predicts the object):

| column | type | range / values | meaning |
|--------|------|----------------|---------|
| `gold_ntokens` | int | `≥ 0` | how many tokens the gold answer spans (see the ⚠️ below) |
| `is_top1` | bool | — | the **first** gold token is the greedy top-1 |
| `gold_rank` | int | `≥ 0` (`-1` if no gold) | rank of the first gold token in the vocab (`0` = top-1) |
| `logit_gap_top` | float / NaN | `≥ 0` (logit units) | max logit − first-gold logit — the "distance" of the correct logit from the top (`0` ⇔ top-1) |
| `answer_margin` | float / NaN | ℝ (logit units) | first-gold logit − runner-up logit (`>0` ⇔ top-1) |
| `gold_prob_first` | float / NaN | `[0, 1]` | prob of the **first** gold token (unconditioned) |
| `top1_prob` | float | `[0, 1]` | prob of the greedy top token |
| `gold_surprisal` | float / NaN | `≥ 0` (nats) | −log p(first gold token) |
| `pred_entropy` | float | `≥ 0` (nats) | entropy of the next-token distribution (how peaked the guess is) |
| `resolution_depth` | int / NaN | `0 … n_layers` | first layer where the logit lens makes the first gold token top-1 |
| `resolution_depth_frac` | float / NaN | `[0, 1]` | `resolution_depth` normalized by layer count |
| `resid_norm_last` | float | `≥ 0` | L2 norm of the final-layer residual at the answer position |
| `mlp_active_frac_mean` | float | `[0, 1]` | mean fraction of MLP neurons with `|activation| > NEURON_EPS`, averaged over layers |
| `mlp_participation_mean` | float | `≥ 1` | **participation ratio** `(Σa²)²/Σa⁴` = effective # of active neurons, averaged over layers (near `1` = concentrated/specialized, high = spread out) |
| `subject_found` | bool | — | the subject's last token was locatable (else hidden-state / probe rows are NaN) |

The same MLP-activity columns are also emitted at the **subject** token with a
`_subj` suffix (`mlp_active_frac_mean_subj`, `mlp_participation_mean_subj`), all
NaN when `subject_found` is false.

> ⚠️ **Multi-token answers.** `is_top1`, `gold_rank`, `logit_gap_top` and
> `answer_margin` describe the single next-token distribution at the answer
> position, so they look at the **first** gold token only. For a two-token answer
> like "New York" they concern " New", not the whole span. Use `gold_prob`
> (length-normalized over all tokens) for the full answer, and filter on
> `gold_ntokens == 1` when you want the first-token metrics to be unambiguous.

**Internal-state per-layer arrays** (`list[float]`, length `n_layers+1`; for
depth-trajectory / geometry EDA, not for the flat modeling table):
`gold_rank_by_layer`, `gold_logit_by_layer`, `resid_norm_by_layer`, and the MLP
per-layer arrays (length `n_layers`, each with a `_subj` twin):
`mlp_active_frac_by_layer`, `mlp_l1_by_layer`, `mlp_participation_by_layer`.

### `hidden_step{N}_{position}.npy` — f16 hidden-state dump (`artifacts/internal/`)

Written unless `--no-hidden`. One memmap-friendly `float16` array per position,
shape `[n_facts, n_layers+1, hidden]` (row `i` ↔ `hidden_step{N}_ids.npy[i]`;
positions that couldn't be located are `NaN`). A sidecar `hidden_step{N}_meta.json`
records shape/positions. Reload lazily with `np.load(path, mmap_mode="r")`.

- **`answer`** *(default)* — the cloze's last token, i.e. the position whose
  next-token prediction *is* the object. This is where all the logit features
  above are read, so it's the natural state for "how/where is the answer formed".
- **`subject_last`** *(default)* — the subject's final token; feeds
  `train_layer_probes` (is the answer linearly readable from the subject?) and
  `representation_drift` (cosine movement of that token between checkpoints).
- **`all`** *(not enabled by default)* — every prompt token, not just one. It
  multiplies the dump by the prompt length (~10–15×, tens of GB per checkpoint),
  so it's left out; add it in the script only if you truly need per-token states.

Size per default position ≈ `n_facts × (n_layers+1) × hidden × 2` bytes
(≈ 68 KB/fact/checkpoint for LMEnt-1B). `config.HIDDEN_POSITIONS` / `--positions`
control which are dumped.

### `neurons_step{N}_{position}.npy` — top-k active-neuron indices (`artifacts/internal/`)

Written **only** with `--top-k K` (`config.NEURON_TOPK`). One `uint16` array per
position (`answer`, `subject_last`), shape `[n_facts, n_layers, k]`, holding the
indices of the k most active MLP neurons per layer (descending `|activation|`).
Row `i` ↔ `neurons_step{N}_ids.npy[i]`; rows for a position that wasn't located
are filled with `65535` (the `missing_sentinel` in `neurons_step{N}_meta.json`,
never a valid index). Unlike the `mlp_*` scalars ("how much" machinery), this is
*which* neurons — it enables the **circuit-overlap / churn** analyses (does a fact
reuse the same neurons across checkpoints, or share them with other facts?). See
[analysis/README.md](analysis/README.md) for recipes.

Size ≈ `n_facts × n_layers × n_positions × k × 2` bytes (~4 KB/fact/checkpoint at
k=64 over both positions — ~3% of the hidden dump).

### `panel.parquet` — one row per (fact × checkpoint) (`feature_extraction/build_feature_table.py`)

Contains `id`, `step`, `correct`, `gold_prob` (above), the **internal-state
scalar columns** at that step (`is_top1`, `gold_rank`, `logit_gap_top`,
`answer_margin`, `pred_entropy`, `resolution_depth`, `mlp_active_frac_mean`, …
whichever were produced by the sweep) **+** all *timeline* and *competition*
columns below, recomputed relative to that checkpoint's `step`.

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

**Internal state at first-learned** (`atlearn_*`, from `internal.extract_fact`) —
the internal-state scalars (above) snapshotted at the fact's `first_learned_step`,
i.e. *how the fact looked in the model the moment it was first acquired*
(`atlearn_logit_gap_top`, `atlearn_resolution_depth`, `atlearn_pred_entropy`,
`atlearn_mlp_active_frac_mean`, …). `NaN` for never-learned facts. These are
**non-leaky** — measured before the end of training — so they're usable as Step C
predictors (the research question: does *how* a fact is first stored predict
whether it's retained?).

> ⚠️ Do **not** substitute the *final*-checkpoint internal state here: because
> `retained` is defined by final-checkpoint correctness, final-step internals
> (e.g. is the answer top-1 at the end) would leak the target, like `leaky_*`.
> `build_feature_table.py` deliberately reads the first-learned step instead.

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
> `DictMentionProvider` backs the tests in `tests/test_mention_features.py`.

**Inside the model** (`features/internal.py`). The per-(fact, checkpoint) internal
features now ship **in `recall_step{N}.parquet`** (via `extract_fact`, documented
above) plus the f16 hidden-state dump. The functions below are the *offline /
notebook* utilities that consume the dump or run extra passes:

| function → column | type | range / values | meaning |
|-------------------|------|----------------|---------|
| `representation_drift` → `drift_cos_mean` | float / NaN | `[-1, 1]` | mean per-layer cosine sim of the `subject_last` state between two checkpoints (feed the dumped arrays) |
| `representation_drift` → `drift_cos_last` | float / NaN | `[-1, 1]` | same at the final layer |
| `train_layer_probes` → (array) | float[] | each `[0, 1]` | per-layer linear-probe accuracy for the answer from the `subject_last` hidden state (offline, CPU) |
| `paraphrase_stability` → `paraphrase_*` | int/float/bool | see code | rephrasing-robustness of the fact — **extra forward passes**, so run it selectively, not in the main sweep |

Single-fact helpers `logit_lens_features`, `subject_hidden_states` and
`_hidden_states` remain for notebook use; the sweep uses `extract_fact` instead so
each fact is only run once.

## ⚠️ Verify against the real download

**Resolved** (from LMEnt / the paper, no longer guessed):
- **Few-shot cloze evaluation, demonstrations held out** — verified against olmes
  commit `1f0568f` (the SHA LMEnt pins): `PopQACloze` defaults to `num_shots=15`,
  as does the registered `popqa_cloze_sitelinks`, and the fork has *no* zero-shot
  cloze config. An earlier note here claimed 15-shot was the benchmark-only
  setting — that was wrong. `config.NUM_SHOTS = 15`, overridable with
  `--num-shots`, and `--num-shots 0` for the bare cloze.
  - `PopQACloze` sets no `fewshot_source` and reports no train/validation split,
    so upstream's `fewshot_examples` falls to `rnd.sample(self.test_docs(), k*2)`
    — **demonstrations sampled from the set being scored**. We instead select a
    held-out pool by the documented rule in `src/fewshot.py`; measured on
    popqa-kas that takes eval facts sharing a demo's answer from 769 to 0.
  - Upstream's own curated `FEWSHOT_SOURCES["popqa"]` is not the default either:
    16 hand-pasted rows with no generating script, one of which has its answer
    inside its question. Available as `--fewshot-source upstream` for parity runs.
  - The pool holds one fact per relation (16), and each fact is shown the pool
    **minus its own relation** — exactly 15 demonstrations, with the relation under
    test never demonstrated. A same-relation demo would hand over the answer *type*
    ("The color of … is *black*"), which is in-context scaffolding for the very
    thing this project measures. Every pool fact still gets used somewhere, unlike
    upstream's `_fewshot_docs[:15]` of 16, which drops the last entry outright so
    `prop_id` 422 is evaluated but demonstrated by nothing.
    `--fewshot-keep-same-prop` restores upstream's single fixed block.
  - Measured on popqa-kas: eval facts that could be answered by copying a demo's
    answer — **0** (ours) vs **773** (upstream). Facts with their gold anywhere in
    the context — 305 / 2.1% (ours) vs 1083 / 7.6% (upstream). The residual 2.1% is
    structural, not a pool defect: a relation's *template* contains words that are
    other relations' answers ("The composer of … is" vs an occupation fact whose
    answer is "composer").
- Prompts, cloze templates (by `prop_id`), exact-match scoring, decode length/stop
  — all from olmes `PopQACloze` via `src/lment_popqa.py`.
- **LR peak `5e-3`, warmup `1000`, batch `32768`** — from the paper §B.4 (the
  authoritative 1B config; `kas_config.json` on `main` is the 170M *example*).
- **1B = 16 layers, 2048 hidden** (§B.4) — validates the internal-features assumptions.
- PopQA-KAS schema (`prop_id`, `possible_answers` as a list, the `*_chunks`
  fields) — confirmed by loading the dataset (see notebook §1).

**Still confirm at checkout** (isolated in `config.py` / tagged `# VERIFY`):
- **`LR_PEAK` superscript** — §B.4 reads `5e-3`; superscripts are easy to misread
  on the PDF, so eyeball it. Low-risk: the LR feature depends on the curve *shape*,
  not absolute scale.
- **`LR_ALPHA_F`** (cosine floor) — §B.4 omits it; OLMo-core's `CosWithWarmup`
  defaults to `0.1`.
- ~~**olmes symbols**~~ — verified against commit `1f0568f`: `PopQACloze` in
  `oe_eval/tasks/oe_eval_tasks/popqa_cloze.py`, `doc_to_text` → `doc["query"]`,
  `doc_to_target` → `" " + gold`, and `Task.fewshot_context`'s layout
  (`"\n\n".join(text + target) + "\n\n" + text`). One template still differs
  deliberately: upstream maps `prop_id` 97 *and* 422 both to `"The capital of {} is"`
  even though 97 is the inverse relation, so we use `"{} is the capital of"` for 97
  — our prompts are therefore **not** byte-identical to upstream for 97's facts.
- **`batch_indices` path & shape** — assumed `dataset-cache/batch_indices.npy`,
  step-indexed array of chunk-ID arrays.
- **Model internals** (`features/internal.py::_final_norm_and_head`) — confirmed
  as `model.model.norm` / `model.lm_head` for HF `Olmo2ForCausalLM`; sanity-check
  with `print(model)` on the real checkpoint.
**Resolved** (from LMEnt's code, no longer guessed):
- **Mention schema** (`features/text.py`) — read from LMEnt's own reader
  (`retrieval-index/create_es_index.py`): per-chunk `metadata["entities"]`, each
  mention having **char** offsets, `text_mention`, and `candidates[{qid,
  aggregated_score, scores_by_source{hyperlinks, entity_linking, coref,
  coref_cluster}}]`. Entities are Wikidata QIDs (`s_uri`/`o_uri`). `mention_features`
  is implemented + unit-tested against this schema; it only *runs* when the 47 GB
  tokenized corpus + `olmo_core` are present (else NaN columns).
