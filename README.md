# Predicting fact retention vs. forgetting in LMEnt

Holding fact *frequency* fixed, which properties of the corpus, the training
schedule, and the model's internal representation predict whether a fact is
**retained** or **forgotten** during pretraining?

Built on the [LMEnt](https://github.com/dhgottesman/LMEnt) suite
([paper](https://arxiv.org/abs/2509.03405)) — the `LMEnt-1B-6E` model (6 epochs,
658,032 steps; `config.MODEL_ID`, override with `LMENT_MODEL_ID`), the
entity-annotated corpus, and the
[`popqa-kas`](https://huggingface.co/datasets/dhgottesman/popqa-kas) dataset.

## Sub-projects

Four sub-projects share this codebase. Each lives in its own directory with a
README (question, headline result, entry point, data it reads):

| directory | question | start here |
|---|---|---|
| [`frequency_forgetting/`](frequency_forgetting/) | does forgetting get *worse* for facts the model sees more often, or is that a counting artefact (count vs. hazard)? | `01_frequency_and_forgetting.ipynb` |
| [`neuron_forgetting/`](neuron_forgetting/) | what forgetting looks like *inside* the model: logit-lens depth, MLP activity, neuron churn | `02_internal_paper_figures.ipynb` |
| [`token_strain/`](token_strain/) | does *token interference* (tokens shared with many other entities) explain forgetting on top of exposure? | `FINDINGS_token_strain.md`, `strain_diagnostics.py` |
| [`explicit_mentions/`](explicit_mentions/) | LMEnt §6.2 intervention: rewrite implicit mentions in the pretraining corpus to explicit names, retrain, compare recall | `README.md`, `analysis/compare_results/` |
| [`legacy/`](legacy/) | the proposal's original panel/facts feature pipeline — superseded, kept for reference | `README.md` |

All four consume artifacts produced by the shared `feature_extraction/` scripts
(GPU) through the shared `src/` library; the notebooks locate the repo root
themselves, so they run from any working directory.

## Layout

**feature_extraction/** runs the model to *produce* artifacts under
`data/artifacts/`; the sub-project directories *consume* them. `src/` is the
shared library both sides import; `config.py` holds every path and constant.

```
config.py                  # all IDs, checkpoint steps, paths, few-shot settings, explicit-mentions constants
src/                       # shared library
  lment_popqa.py           # SINGLE home for the olmes coupling: cloze prompts (by prop_id),
                           #   few-shot context, exact-match scoring, generation kwargs
  data.py                  # load dataset + checkpoints (HF release or local converted dir) + tokenizer
  recall.py                # per-checkpoint cloze recall (olmes scoring) + gold-answer prob
  fewshot.py               # held-out demonstration pool + few-shot modes (drop/only-same-prop)
  fewshot_source.py        # LMEnt's verbatim demonstrations (for --fewshot-source upstream)
  prompt_set.py            # frozen prompt parquets: build once, score every model on identical strings
  features/
    internal.py            # one-pass extractor: logit-lens depth, next-token geometry,
                           #   MLP activity, hidden-state + top-k neuron dumps      [neuron_forgetting]
    strain.py              # token interference: EntityUniverse, TokenIndex (strain, ownership),
                           #   alt-answer co-occurrence, WindowMap over all 6 epochs [token_strain]
feature_extraction/        # PRODUCES artifacts
  build_prompts.py         # Step A.0: freeze the few-shot prompts       -> data/artifacts/prompts/
  eval_checkpoints.py      # Step A.1: recall + internal-state sweep   -> data/artifacts/recall/ (+ internal/)
                           #   --model may be an HF release or a local dir of converted checkpoints
  build_strain_panel.py    # (fact x window) interference panel        -> data/artifacts/features/  [CPU]
tests/                     # test_strain_features, test_neuron_dump, test_fewshot_internals
docs/                      # the LMEnt paper (+ Gottesman et al. final version), the proposal, project.bib
data/artifacts/            # committed results: recall/ features/ expermints/ prompts/ popqa/
                           #   bulk outputs data/{corpora,hf,runs} (~940 GB) are gitignored and live on the
                           #   lab filesystem -- see explicit_mentions/README.md "Where the bulk data lives"

frequency_forgetting/      # sub-project: notebooks + the two trajectory CSVs
neuron_forgetting/         # sub-project: notebook + the internal-features docs
token_strain/              # sub-project: strain_models / strain_diagnostics / report / null control / FINDINGS
explicit_mentions/         # sub-project: corpus rewrite -> OLMo-core training -> HF conversion -> slurm,
  analysis/                #   plus its notebooks (corpus/model comparisons, trajectories, Fig. 9)
legacy/                    # superseded panel/facts pipeline (build_feature_table, schedule, 4 feature modules)
```

## Shared feature modules

| Module | Used by | Notes |
|--------|---------|-------|
| `src/features/internal.py` | `neuron_forgetting/` | **one forward pass per (fact, checkpoint)**: logit-lens depth, next-token geometry (margin, top-logit gap, rank, entropy), per-layer residual/MLP-neuron activity (active fraction + participation ratio, at answer & subject), plus f16 hidden-state and optional top-k neuron-index dumps for offline probe / drift / circuit-overlap | |
| `src/features/strain.py` | `token_strain/` | *token* interference, as opposed to the chunk-level interference in `legacy/`: how many **other entities** share a token with this fact's subject / object (`strain`), what share of the evidence carrying those tokens is this fact's (`ownership`), and how often the subject co-occurs with a *different* valid answer for the same relation (`alt-answer`). Static and per-window forms. Requires no GPU | |

## Setup

```bash
pip install -r requirements.txt
export LMENT_DATA_ROOT=/path/to/storage      # checkpoints, corpus, artifacts
```

Download from HuggingFace: the model `dhgottesman/LMEnt-1B-6E` (checkpoints are
loaded on demand by subfolder) and the `LMEnt-Dataset` pieces you need —
`dataset-cache` at minimum; `dataset-metadata` (mentions) only for the
text-mention features.

**Get all six per-epoch batch-index files, not just `batch_indices.npy`:**

```
dataset-cache/batch_indices_epoch_1.npy   ... _epoch_6.npy    # 90 MB each, 540 MB total
```

`batch_indices.npy` is byte-identical to `batch_indices_epoch_1.npy` and covers
epoch 1 only, which is what LMEnt's own README documents. OLMo-core reshuffles
every epoch (`data_loader.py`: `get_rng(seed + epoch + seq_len)`), so a chunk sits
at a different step in each — and **epoch 1 holds only ~5% of the forgetting
events**, since accuracy climbs monotonically through it. `build_strain_panel.py`
requires all six and fails with a clear message if any are absent.

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
# Step A.0 (optional but recommended) — freeze the few-shot prompts
python feature_extraction/build_prompts.py --num-shots 15
#   Renders the prompts ONCE to data/artifacts/prompts/prompts{tag}.parquet: the exact prompt
#   string per fact, which facts are evaluated, in which order, how many demos each got
#   (`num_shots` is per fact) and which ones (`demo_ids`). The flags, the demonstration
#   pool and a `prompt_hash` over the strings are stored in the parquet's schema metadata
#   (mirrored to a `*.meta.json` sidecar); `--inspect FILE` prints them back.
#   Takes the same --num-shots / --fewshot-mode / --fewshot-source / --fewshot-per-prop /
#   --min-shared-chunks / --limit flags as the eval, plus --out / --out-dir / --force /
#   --show N. Pass the file to the eval with --prompts and every model, checkpoint and
#   rerun is scored on identical inputs, even if the selection rule or the dataset later
#   changes -- which is what makes model-vs-model numbers comparable.
# -> data/artifacts/prompts/prompts_15shot.parquet (+ .meta.json)

# Step A.1 — recall + internal states at every checkpoint (GPU; one per checkpoint)
python feature_extraction/eval_checkpoints.py --prompts data/artifacts/prompts/prompts_15shot.parquet
python feature_extraction/eval_checkpoints.py            # or --steps 0 10000 ... as a Slurm array
#   --prompts FILE         use a pre-built prompt file (Step A.0) instead of rendering the
#                          prompts from the flags below; the file then defines the prompts,
#                          the eval set and its order, and supplies the output filename tag,
#                          so a run off the same file is byte-identical to one off the
#                          matching flags. Incompatible with --num-shots / --fewshot-* /
#                          --min-shared-chunks (those were settled when the file was built).
#   --num-shots N          in-context demos (default 15; 0 = bare cloze). Demos come from a
#                          held-out pool of one fact per relation, never the eval set, and a
#                          fact is never shown its OWN relation -- hence 15 of 16. The pool's
#                          16 subjects are dropped at EVERY shot count, so 0-shot and n-shot
#                          score an identical fact set. Output is suffixed `_{N}shot` so runs
#                          at different N don't overwrite each other.
#   --fewshot-mode M       which demonstrations a fact sees -- the knob for comparing
#                          few-shot methods. drop-same-prop (default) = the pool minus the
#                          query's own relation; fixed-block = one constant block incl. its
#                          own relation (upstream, pair with --num-shots 16);
#                          only-same-prop = ONLY its own relation, with every demo whose
#                          answer would be scored as the query's answer removed. That mode
#                          selects its own bigger pool (num-shots+5 per relation) and so
#                          holds out ~305 eval facts instead of 16 -- compare it to the
#                          other modes on the id intersection. Output suffix `_onlysameprop`.
#   --fewshot-keep-same-prop    deprecated alias for --fewshot-mode fixed-block
#   --fewshot-source upstream   use LMEnt's verbatim demos (leaky; parity runs only)
#   --no-hidden            skip the f16 hidden-state dump (scalar features only)
#   --internal-fewshot     measure the internal features on the same few-shot prompt the
#                          greedy decode saw. OFF by default: the internals (gold_prob,
#                          logit lens, hidden states, neurons) are otherwise read from the
#                          bare ZERO-SHOT cloze even in an n-shot run, so the answer
#                          position is unconditioned and comparable across shot counts --
#                          at the cost of `correct` and the internals living on two
#                          different prompts. Turning it on makes them agree but changes
#                          what the depth profile describes: retrieval under in-context
#                          conditioning rather than how the fact is stored. Output is
#                          suffixed `_fsinternal` so it never collides with the
#                          zero-shot-internals artifacts.
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
# -> data/artifacts/recall/recall_step{N}.parquet   (recall + internal scalar/per-layer features)
# -> data/artifacts/internal/hidden_step{N}_*.npy   (f16 [n_facts, n_layers+1, hidden] per position)
# -> data/artifacts/internal/neurons_step{N}_*.npy  (uint16 [n_facts, n_layers, k]; only with --top-k)


# Step B — the analyses live in the sub-project directories; each README says what
#          it reads and how to regenerate it:
#   token_strain/README.md        build_strain_panel.py -> strain_diagnostics.py -> report_token_strain.py
#   frequency_forgetting/README.md, neuron_forgetting/README.md   notebooks on the recall sweeps
#   explicit_mentions/README.md   corpus rewrite -> train -> convert -> eval_checkpoints.py --model <local dir>
```

**One inference per (fact, checkpoint).** `eval_checkpoints.py` touches the model
exactly twice per fact: once for the greedy decode `correct` needs, and once for a
teacher-forced forward that yields *everything else* — `gold_prob`, logit-lens
depth, next-token geometry, per-layer residual/MLP activity, and the hidden-state
dump. (A true single pass is possible only if you redefine `correct` as
"first answer token is top-1", i.e. the `is_top1` column — but that changes the
paper's metric, so the decode is kept.)

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
| `step` | int | one of `CHECKPOINT_STEPS` (0 … 658032) | checkpoint evaluated |
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
(≈ 76 KB/fact/checkpoint for LMEnt-1B: 19 × 2048 × 2). `config.HIDDEN_POSITIONS` / `--positions`
control which are dumped.

### `neurons_step{N}_{position}.npy` — top-k active-neuron indices (`artifacts/internal/`)

Written **only** with `--top-k K` (`config.NEURON_TOPK`). One `uint16` array per
position (`answer`, `subject_last`), shape `[n_facts, n_layers, k]`, holding the
indices of the k most active MLP neurons per layer (descending `|activation|`).
Row `i` ↔ `neurons_step{N}_ids.npy[i]`; rows for a position that wasn't located
are filled with `65535` (the `missing_sentinel` in `neurons_step{N}_meta.json`,
never a valid index). Unlike the `mlp_*` scalars ("how much" machinery), this is
*which* neurons — it enables the **circuit-overlap / churn** analyses (does a fact
reuse the same neurons across checkpoints, or share them with other facts?).

Size ≈ `n_facts × n_layers × n_positions × k × 2` bytes (~4 KB/fact/checkpoint at
k=64 over both positions — ~3% of the hidden dump).

### `strain_panel.parquet` — one row per (fact × window) (`feature_extraction/build_strain_panel.py`)

A **different shape** from the legacy per-checkpoint panel: rows exist only for windows where the
fact was **correct at the window's start**, so the outcome `forgot` is defined on a
population that could actually produce the event (a discrete-time hazard design).
Window `w` spans `(steps[w], steps[w+1]]`, taken from the trajectory's own header.

| column | type | meaning |
|--------|------|---------|
| `id` / `prop` / `window` / `step_lo` / `step_hi` | — | fact, relation, window index and its step bounds |
| `forgot` | int | **outcome.** 1 if correct at `step_lo` and wrong at `step_hi` |
| `subj_strain` / `obj_strain` | float | how many **other entities** (QIDs) share ≥1 token with the subject / object. The **size of the union** of rival sets over the entity's tokens — not a sum, not a mean — excluding the fact's own other endpoint |
| `subj_strain_w` / `obj_strain_w` | float | the same, restricted to entities active in *this* window |
| `evidence_static` / `evidence_w` | int | **exposure**: chunks where the subject co-occurs with the true answer *or* a rival answer of the same relation (`shared + distract`), total and per-window |
| `shared_static` / `shared_w` | int | chunks pairing the subject with the **true** answer |
| `distract_static` / `distract_w` | int | chunks pairing the subject with a **different** valid answer for that relation |
| `ratio_static` / `ratio_w` | float | **F2**: `(shared+1)/(shared+distract+2)`, Laplace-smoothed. Smoothing is not cosmetic — unsmoothed, a fact with `shared=1, distract=0` scores a perfect 1.0, which fills the "cleanest" bucket with no-evidence facts |
| `s_own_rare` / `o_own_rare` | float | **F1**: `num_shared_chunks ÷ |all chunks carrying the entity's rarest token|` |
| `s_rivals_rare` / `o_rivals_rare` | int | **F1**: how many other entities carry that rarest token |
| `s_ntok` / `o_ntok` | int | **name length**, in post-cutoff tokens |
| `s_charlen` / `o_charlen` | int | name length in characters. Correlates with `s_ntok` at spearman 0.88 — kept as a robustness check, **not** for joint use with it |
| `s_pop` / `o_pop` / `num_shared_chunks` | int | carried through from `popqa-kas` |

> **Name length is not optional.** Strain rises steeply with how many tokens a name
> has (mean strain 4.8 / 28 / 72 / 109 / 127 for 1–5 token names, spearman 0.61),
> so on a broad fact set the strain terms are largely a proxy for it — and
> controlling it reverses their sign. Condition on `s_ntok`/`o_ntok` in any model,
> and put them in the baseline of any permutation test.

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
    `--fewshot-mode fixed-block` restores upstream's single fixed block.
  - **`--fewshot-mode only-same-prop`** is the deliberate opposite, so the value of
    that answer-type information can be measured instead of assumed: every demo
    shares the query's relation, and every demo whose answer would be *scored* as the
    query's answer is dropped (`fewshot.answers_clash`, olmes' substring rule in both
    directions plus shared `obj_id`). Type for free, copying impossible. It needs
    ~20 demos per relation rather than one, so the global answer-uniqueness filter is
    replaced by that per-fact rule (see the `SAME_PROP_*` comments in
    `src/fewshot.py`) and the held-out pool grows to 305 facts — its eval set is a
    subset of the other modes', so compare on the `id` intersection.
    Measured on popqa-kas at `--num-shots 15`: facts answerable by copying a demo's
    answer **0**, gold string anywhere in context **144 / 1.0%** (vs 2.1% for
    `drop-same-prop`), and 13,733 of 13,959 facts get the full 15 demos. The
    remainder are relations with a narrow answer range — most sharply prop 472 ("The
    color of X is", four distinct answers among all candidates, 21 of 24 being "red"),
    whose facts get 3–4. Demos per answer are capped at 2 for exactly that reason: an
    uncapped colour bench is 17× "red", which is a legal block and a useless one. The
    per-fact count is recorded in the output's `num_shots` column.
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
- **1B = 18 layers, 2048 hidden, `d_ff` 5632** — read off the checkpoint `config.json`
  (`num_hidden_layers: 18`, Olmo2), which is authoritative over any paper-table reading;
  validates the internal-features assumptions.
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
- ~~**`batch_indices` path & shape**~~ — verified on download. `dataset-cache/
  batch_indices_epoch_{1..6}.npy`, each an object array of 109,672 steps × 16
  chunk IDs (6 × 109,672 = 658,032 = the final 6E checkpoint). `batch_indices.npy`
  is byte-identical to `_epoch_1`; all six orderings differ (chunk 3561491 lands at
  local step 0 / 80374 / 53637 / 27332 / 24924 / 68259). Seed 0, confirmed by the
  sibling `vsl-grow-p2-8-cycle/global_batch_indices_bz*_epoch{1..6}_seed0.npy`.
- **Model internals** (`src/features/internal.py::_final_norm_and_head`) — confirmed
  as `model.model.norm` / `model.lm_head` for HF `Olmo2ForCausalLM`; sanity-check
  with `print(model)` on the real checkpoint.
