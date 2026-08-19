# Explicit-Mentions Experiment

Implements the causal intervention proposed in the LMEnt paper (§6.2, "Entity
Mention Forms and Learning"): **replace implicit entity mentions (pronouns,
aliases, descriptors found only by coreference) in the pretraining corpus with
the entity's explicit canonical name, retrain, and measure whether factual
learning improves.**

This directory is self-contained (nothing here imports the parent repo). It
reads the LMEnt corpus in place from the authors' lab directory
(`/home/morg/students/gottesman3/LMEnt`, read-only) and reuses their OLMo-core
fork as a library for training — all outputs go to `EXPLICIT_ROOT`
(default `/home/morg/NLP_2526b/shaharmayo/explicit_mentions_data`, override
via env var).

## What the input looks like

One row of `dataset-cache/dataset-metadata/part-N-00000.csv` = one Wikipedia
document. Columns: `start,end,id,src,loc,title,entities,offsets` where
`start`/`end` are token offsets into the sibling `.npy`, `entities` is a JSON
list of mentions, and `offsets` maps each token to its character span.

A single mention, verbatim:

```json
{"char_start": 243, "char_end": 253, "text_mention": "the boat's",
 "candidates": [{"qid": "Q28232329", "name": "Shellbend",
   "scores_by_source": {"hyperlinks": 0.0, "entity_linking": 0.0,
                        "coref": 0.0, "coref_cluster": 1.0},
   "aggregated_score": 0.1}]}
```

`scores_by_source` is what drives everything: `hyperlinks` or `entity_linking`
> 0 means the mention names the entity **explicitly** (left untouched); only
`coref`/`coref_cluster` means it was found **implicitly** (rewritten). Note
mentions overlap heavily — the same text is often annotated at several spans
with different sources — which is why the pipeline resolves overlaps greedily.

## How the rewrite works

Every document in the corpus metadata carries entity mentions:
`char_start/char_end` spans, each with candidate QIDs scored per source
(`hyperlinks`, `entity_linking`, `coref`, `coref_cluster`). A mention is
**implicit** when its best candidate has *zero* hyperlink and entity-linking
evidence — it was found only by coreference (e.g. *"he"*, *"the team"*,
*"the hospital"*). For each implicit mention with coref score ≥ 0.6 (the
paper's own retrieval threshold), `build_explicit_corpus.py`:

1. decodes the document's tokens back to text (spans are validated against
   every mention's `text_mention` before anything is touched);
2. substitutes the candidate's canonical `name` (possessive pronouns like
   *his/their* become *"<Name>'s"*);
3. remaps every mention's char span, re-tokenizes with the dolma2 tokenizer,
   recomputes token spans (`tok_start/tok_end`, required by OLMo-core's
   entity-aware chunking), and re-emits the document.

The output corpus has the exact on-disk layout OLMo-core's `kas_vsl` dataset
expects, so the authors' training pipeline consumes it unchanged (VSL bucket
caches are rebuilt automatically in our own `dataset-cache/`). Rewritten
mentions keep their candidates and gain a `rewritten_from` field, so
post-training attribution analyses still work.

### Experiment axes (flags)

| Experiment | Flags |
|---|---|
| Basic: replace **all** implicit mentions | *(defaults)* `--corpus-tag explicit_all` |
| Only PopQA-relevant docs / entities | `--scope popqa` (skip other docs) / `--target-qids popqa` (only rewrite mentions of PopQA subjects+objects) |
| Partial replacement (additive effect) | `--replace-fraction 0.5` (deterministic per `--seed`) |
| No pronouns (aliases/descriptors only) | `--no-pronouns` |
| Exhaustive rewriting (ablation) | `--allow-repeat-in-sentence` |

### Why "coref-only" alone is not enough

`hyperlinks == 0 and entity_linking == 0` is the right *starting* filter, but it
over-fires badly. Coreference annotates spans that overlap already-linked
mentions, spans far longer than a name, first-person pronouns inside
quotations, and clusters whose "canonical name" is a common noun. Eight guards
(all ON by default, all logged to `skipped_part-*.jsonl` via `--log-skipped`)
remove those cases. Each was found by reviewing real output
(`inspect_rewrites.py review`), not anticipated:

| guard | what it blocks | example |
|---|---|---|
| `contains_name` | surface already spells out every word of the name | `"the Shellbend"` -> `"Shellbend"` (only deletes *the*) |
| `too_long` | span over `--max-mention-words` (4): appositions/relative clauses whose replacement **deletes facts** | `"her mother Winifred (née Reid)"` -> `"Mary Catherine Tinney"` |
| `name_in_context` | surface is a name fragment and the surrounding text supplies the rest | `"...the 1953 Big Ten Conference football season"` would double to `"the 1953 1953 Big Ten Conference football season football season"` |
| `person_pronoun` | 1st/2nd person pronouns — nearly always inside a quotation | `"I've had a wonderful time"` -> `"Carrie Keagan've had..."` |
| `relative_pronoun` | who/whom/whose/which/that — leaves a comma splice | `"Keagan, who departed"` -> `"Keagan, Carrie Keagan departed"` |
| `name_not_proper` | the entity's name has no capital letter, so it is a common noun | `"they"` -> `"post office"` ("until post office moved") |
| `contradicted` | a **hyperlink or confident entity-link names that same span/surface as a different entity** — a stronger source than coref, so coref loses | `"France"` -> `"Prime Minister of France"`, `"Japan"` -> `"COVID-19 pandemic in Japan"` (EL says the span is Q142 / Q17) |
| `name_guard` | bare proper name with no lexical relation to the target (coref misattribution) | `"California"` -> `"Los Angeles"`, `"Bear Mountain"` -> `"Fresno County"` |
| `multiline` | span crosses a line break (article title fused to first body mention) | |
| `sentence_has_explicit` | the entity is **already named explicitly in that sentence**, so the fact is already learnable there | `"Keagan announced her decision"` -> `"Keagan announced Carrie Keagan's decision"` |
| `sentence_duplicate` | a second implicit mention of the same entity in one sentence — the first was already made explicit | `"Ricardo Moyano has a channel on which Ricardo Moyano uploads"` |

`contradicted` is the only guard that uses evidence rather than a language
heuristic, and it catches a class the others miss: surface and name *share* a
word, so `name_guard` exempts them, yet the coref claim is a containment error
that would turn *"founded in France"* into *"founded in Prime Minister of
France"*. Roughly 4% of otherwise-eligible rewrites are vetoed this way.

`name_guard` also stands down when a stronger source **agrees** with the coref
claim: if a hyperlink or entity-link says that same surface names that same
entity anywhere in the document, it cannot be a misattribution, so vouched-for
aliases and nicknames survive — `"Rini"` -> `"Chibiusa"`, `"Leningrad"` ->
`"Saint Petersburg"`, `"Pegu"` -> `"Bago"`. That rescues 4.4% of what the guard
would otherwise block (counter `name_guard_waived`). An unvouched nickname with
no lexical overlap (`"Thunderbolt"` -> `"Frederick Ward"`) is still blocked.

`name_guard` is otherwise the weakest link — a capitalization/determiner heuristic. It
tests *lexical relatedness* generously (shared word, shared stem `"Ron"` /
`"Ronald Taylor"`, or near-identical spelling `"Toolsboro"` / `"Toolesboro"`)
so short forms survive, but it will still block a determiner-less alias with no
character overlap (`"Queen City"` -> `"Buffalo"`). It deliberately exempts
*descriptions* — anything determiner-led,
possessive-led, or starting lowercase — so the paper's own case still rewrites:
`"the president"` and `"the 47th president"` both become `"Donald Trump"`. And
*partial* names still rewrite, since they genuinely add the missing words:
`"the Bills"` -> `"Buffalo Bills"`, `"Allen"` -> `"Josh Allen"`, `"NISH"` ->
`"National Institute of Speech and Hearing"`.

**One explicit name per entity per sentence.** The last two guards deserve
emphasis: 49% of otherwise-eligible rewrites share a sentence with another
mention of the same entity (18.9% with an already-*explicit* one, 28.2% a
second implicit one). Rewriting those adds no explicitness where the fact
occurs and produces text no English speaker would write. The default keeps the
*first* implicit mention per entity per sentence and leaves the rest, so
*"He has a YouTube channel on which he uploads"* becomes *"Ricardo Moyano has a
YouTube channel on which he uploads"* — explicit once, still fluent.
`--allow-repeat-in-sentence` restores the exhaustive behaviour for an ablation.

Measured on 20 K documents of part-0 — of 293 K eligible implicit mentions,
**37.3% are rewritten**. Held back: `sentence_has_explicit` 17.8%,
`sentence_duplicate` 11.0%, `too_long` 10.6%, `contains_name` 10.0%,
`low_score` 10.0%, `name_not_proper` 4.2%, `multiline` 3.4%, `name_guard` 3.1%,
`name_in_context` 1.2%, `person_pronoun` 0.9%. The corpus grows **+4.2%** in
tokens (was +7.7%) and mention-annotation retention is **99.7%** (88.5% before
any guards existed).

### Reviewing rewrite quality

```bash
python inspect_rewrites.py review --part 0 --n-docs 400 --skip 300000 \
    --per-category 6 --out rewrite_review.md
```

Samples documents anywhere in a part (`--skip`), applies the real selection
logic, and writes a markdown report of before/after **sentences** grouped by
mention type (pronoun, possessive pronoun, description, partial name, ...)
plus what each guard held back. Run this before committing to a full rewrite —
every bug above was found this way.

### Known residual noise

Two things the guards cannot fix, both visible in a review run:

* **Missing determiners.** `"in contempt of parliament"` ->
  `"in contempt of New South Wales Legislative Assembly"` reads better with
  *the*. Detecting this needs a grammar model.
* **Coref misattribution that is lexically invisible.** `"at year's end"` ->
  `"at 2019 BWF World Tour Finals's end"` — the cluster is simply wrong, and
  no surface cue reveals it. Raising `--min-coref-score` above 0.6 trades
  recall for fewer of these.

Unnatural repetition *within* a sentence is now handled by the one-name-per-
sentence rule above; repetition *across* consecutive sentences remains
("*Ricardo Moyano lives in Istanbul. Ricardo Moyano has a YouTube channel...*"),
which is inherent to what §6.2 proposes. `--replace-fraction` dials it back.

### Crash safety / resume

Progress is checkpointed every `--checkpoint-every` documents (default 20 000):
both output files are flushed and `fsync`ed, then the byte offsets are recorded
atomically to `progress_part-N.json`. **Rerunning the identical command
resumes** — it truncates any unrecorded tail left by the kill and continues
from the next document, seeking the input in O(1) via the authors' per-part
line index. This is verified: a `SIGKILL`ed run that left 283 KB of dirty tail
data, then resumed, produced **byte-identical** `.npy` and `.csv` output and
identical statistics to an uninterrupted run.

* A completed part is a no-op on rerun (`--restart` forces a rebuild).
* Resuming with *different* rewrite flags is refused rather than splicing two
  different corpora together — use a new `--corpus-tag` or `--restart`.
* So on the preemptible partitions, just resubmit the same sbatch command.

Two-phase training ("1 epoch original, then explicit") is a *training* flag:
`make_train_config.py --load-path <phase1 checkpoint>`.

## Pipeline

```
build_explicit_corpus.py  ->  inspect_rewrites.py  ->  make_train_config.py +
   (slurm array, CPU)          (verify + eyeball)      train_explicit.py (GPU)
                                                              |
eval_explicit.py  <-  convert_checkpoints.py  <---------------+
   (GPU, PopQA)          (OLMo-core -> HF)
      ^
build_prompts.py   (once: the frozen prompt file every model is scored on)
```

### 0. Environment

Commands below use `$EXPLICIT_ROOT` for the output root. It is **not** set for
you — either substitute the literal path
`/home/morg/NLP_2526b/shaharmayo/explicit_mentions_data`, or set it in your
shell first (this lab's default shell is tcsh):

```tcsh
setenv EXPLICIT_ROOT /home/morg/NLP_2526b/shaharmayo/explicit_mentions_data
```
```bash
export EXPLICIT_ROOT=/home/morg/NLP_2526b/shaharmayo/explicit_mentions_data   # bash/zsh
```

`make_train_config.py` prints absolute paths in the commands it suggests, so
copying from its output avoids the variable entirely.


* Rewriting + eval run in the existing **`esm_env`** conda env.
* Training + checkpoint conversion need OLMo-core's deps in a new env (once):

```bash
conda create -n lment_train python=3.12 -y
conda activate lment_train
pip install "numpy<2.0" "torch>=2.5.1" cached-path requests packaging rich \
    omegaconf safetensors importlib_resources ai2-olmo-eval==0.5.0 \
    transformers datasets wandb
# OLMo-core itself is imported from the authors' checkout via sys.path —
# no pip install needed, nothing is written into their directory.
```

### 1. Smoke test (minutes, can run on a login node)

```bash
conda activate esm_env
cd explicit_mentions
python build_explicit_corpus.py --part 0 --limit 2000 --corpus-tag smoke --samples 100
python inspect_rewrites.py samples --corpus-tag smoke --n 20   # eyeball quality
python inspect_rewrites.py verify  --corpus-tag smoke --part 0 # structural checks
```

### 2. Full rewrite (slurm, 8 CPU array tasks, hours each)

```bash
mkdir -p logs
sbatch slurm/rewrite_corpus.sbatch --corpus-tag explicit_all
# preempted / crashed? resubmit the same line -- each part resumes itself:
sbatch slurm/rewrite_corpus.sbatch --corpus-tag explicit_all
# then verify each part:
for p in 0 1 2 3 4 5 6 7; do
    python inspect_rewrites.py verify --corpus-tag explicit_all --part $p --n-docs 20000
done
# read the per-part stats + audit the guards:
cat $EXPLICIT_ROOT/corpora/explicit_all/stats_part-*.json
head $EXPLICIT_ROOT/corpora/explicit_all/skipped_part-0-00000.jsonl
```

### Cost (measured, single core per part)

The corpus is 6.79 M documents / 3.80 B tokens across 8 parts, very unevenly
sized (part-2 alone is 1.02 B tokens — 27% of the corpus — so it sets the
wall-clock of the array job). Throughput scales with *tokens*, measured at
~110 k tok/s for a full rewrite, sampled from four offsets in parts 0 and 2:

| mode | throughput | wall clock (8 parts in parallel) | CPU-hours |
|---|---|---|---|
| full rewrite (default) | 110 k tok/s | **≈ 2.8 h** | ≈ 10 |
| `--scope popqa` | 166 k tok/s | **≈ 1.8 h** | ≈ 6.4 |
| `--target-qids popqa` | 246 k tok/s | **≈ 1.2 h** | ≈ 4.3 |

Wall clock = part-2's time plus ~8 min to gzip its metadata (gzip runs at
~100 MB/s, compressing 6.5:1). Every part finishes well inside the 4-day
`cpu-killable` limit, so preemption is the only realistic interruption — and
that resumes.

`--scope popqa` saves less than you might expect (1.5×, not 10×): PopQA's
19 384 entities include very common ones (countries, large cities), so ~70% of
all Wikipedia documents still mention at least one and are processed anyway.
`--target-qids popqa` is the cheaper and much more surgical option — on a
part-0 sample it rewrote 4.6 K mentions versus 86 K for the full rewrite.

Disk: ~215 GB metadata CSV + ~32 GB `.csv.gz` + ~15 GB tokens ≈ **265 GB**
(56 bytes of CSV per token). Delete `dataset-cache/dataset-metadata/*.csv` only
after training — OLMo-core reads the decompressed CSVs, and only the `.csv.gz`
is needed to rebuild them.

### 2b. Pre-build the dataset cache (slurm, CPU, hours)

The first time OLMo-core opens a corpus it indexes every metadata CSV (~215 GB
of reads), scans for document boundaries, computes entity-safe chunk
boundaries, and builds the VSL buckets. By default that happens *inside* the
first training job, with the GPUs idle. Do it on CPU instead:

```bash
python make_train_config.py --run-name explicit_170M_1E --corpus explicit_all
sbatch slurm/prebuild_cache.sbatch $EXPLICIT_ROOT/runs/explicit_170M_1E/config.json
```

It writes into the corpus's own `dataset-cache/`, keyed by a hash of the
dataset settings — so it must use the *same* config.json the training run will
use, which is why it reuses `train_explicit.build_config()`. The training job
then reuses the cache and starts stepping immediately. Skip this only if the
config's `paths`/`max_sequence_length`/`vsl_curriculum` will differ.

### 3. Train (slurm, GPU)

The compute-optimal setup from the paper is **170M for 1 epoch** — and the
published `dhgottesman/LMEnt-170M-1E` is the natural baseline. For a clean
comparison you can also train your own control on the original corpus with
identical seeds/config.

Defaults follow the paper (§B.4): AdamW, global batch 32 768, rank batch 8 192,
**peak LR 5e-3**, **weight decay 0.05**, 1 000 warmup steps.

**Only the final model is kept.** No permanent intermediate checkpoints are
written (not even step0); a single *rolling* ephemeral checkpoint is saved every
`--ephemeral-save-interval` steps (default 1 000) purely so a preempted job can
resume, and it is replaced each time rather than accumulating. The trainer
always writes the final model when training completes. Pass
`--save-interval 10000` if you later want intermediates for a learning-dynamics
sweep.

```bash
# explicit corpus:
python make_train_config.py --run-name explicit_170M_1E --corpus explicit_all
sbatch slurm/train.sbatch $EXPLICIT_ROOT/runs/explicit_170M_1E/config.json
# optional control on the original corpus:
python make_train_config.py --run-name original_170M_1E --corpus original
sbatch slurm/train.sbatch $EXPLICIT_ROOT/runs/original_170M_1E/config.json
```

The killable partition's 24 h limit is shorter than an epoch: **resubmit the
same sbatch command** and the trainer resumes from its last checkpoint.

Two-phase schedule (original then explicit):

```bash
python make_train_config.py --run-name phase2_170M --corpus explicit_all \
    --load-path $EXPLICIT_ROOT/runs/original_170M_1E/olmo2_170M_*/step109000
```

### 4. Convert checkpoints to HF format

```bash
sbatch slurm/convert_to_hf.sbatch \
    --run-dir $EXPLICIT_ROOT/runs/explicit_170M_1E/olmo2_170M_0.0003_32768_0.01_1 \
    --name explicit_170M_1E
```

### 5. Evaluate PopQA recall

Same prompts/scoring as the main project (olmes cloze, held-out few-shot
pool), so results are comparable with existing artifacts.

Build the prompts **once** first and hand the file to every model. The
comparison here is model-vs-model, so the prompts must not be re-derived per
run: the file fixes the prompt strings, the fact set and its order, and carries
a `prompt_hash` you can compare across runs to prove they matched (see
`emlib/prompt_set.py`).

```bash
python build_prompts.py --num-shots 5            # mode defaults to only-same-prop
# -> $EXPLICIT_ROOT/artifacts/prompts/prompts_5shot_onlysameprop.parquet (+ .meta.json)
python build_prompts.py --inspect $EXPLICIT_ROOT/artifacts/prompts/prompts_5shot_onlysameprop.parquet
```

```bash
setenv P $EXPLICIT_ROOT/artifacts/prompts/prompts_5shot_onlysameprop.parquet
sbatch slurm/eval.sbatch --model $EXPLICIT_ROOT/hf/explicit_170M_1E \
    --run-tag explicit_170M_1E --prompts $P
# baseline (published model):
sbatch slurm/eval.sbatch --model dhgottesman/LMEnt-170M-1E --steps 109672 \
    --run-tag baseline_170M_1E --prompts $P
```

Without `--prompts` the prompts are still rendered from `--num-shots` /
`--fewshot-mode` at run time, exactly as before (`--prompts` is incompatible
with those flags — the file already answers them).

Outputs land in `$EXPLICIT_ROOT/artifacts/<run-tag>/recall_step{N}_*.parquet`
with the same schema as the main repo's recall parquets
(`id, step, num_shots, correct, excluded, prompt, generation`), so the
existing analysis notebooks apply. The headline comparison: accuracy binned by
the fact's `num_shared_chunks` (and, per §6.2, by the fraction of explicit
co-mentions in the *original* data — facts that used to co-occur implicitly
are where the explicit corpus should help most).

## Files

| File | Purpose |
|---|---|
| `config.py` | all paths/constants; env-var overridable |
| `build_explicit_corpus.py` | **the mentions script** — rewrites a corpus part |
| `inspect_rewrites.py` | before/after samples + structural verification |
| `train_explicit.py` | OLMo-core training (patched copy of authors' `kas/train.py`: config-controlled wandb/eval, `load_path` for two-phase) |
| `make_train_config.py` | generates training config JSONs |
| `convert_checkpoints.py` | OLMo-core checkpoints → HF `stepN/` layout |
| `build_prompts.py` | renders the few-shot prompts once to a parquet (the eval's `--prompts` input), so every model is scored on identical inputs |
| `eval_explicit.py` | PopQA recall sweep (recall-only port of the main repo's `eval_checkpoints.py`) |
| `emlib/` | copies of the main repo's prompt/recall/few-shot modules (self-contained) |
| `prebuild_dataset_cache.py` | builds OLMo-core's index/chunk/VSL caches on CPU before training |
| `slurm/` | sbatch wrappers for each stage |

## Known caveats

* Grammar around replaced pronouns can be awkward ("*led the Bills to
  Buffalo Bills's first title*") — that is inherent to the intervention; use
  `--no-pronouns` for a conservative variant.
* ~2.5% of *mention annotations* don't survive a rewrite (97.5% retained on the
  20K-doc sample): the corpus annotates many nested/duplicate spans over the
  same text, and a span whose boundary lands inside a replaced region is
  dropped. This does not affect training (annotations are never shown to the
  model) — only post-hoc retrieval/attribution over the rewritten corpus.
* Documents whose decoded text fails span validation are passed through
  unchanged and counted in `stats_*.json` (`docs_decode_mismatch`); expect ~0.
* `make_train_config.py --corpus original` points OLMo-core's `work_dir` at
  the authors' prebuilt cache (read-only, already complete). Corpora built
  here get their own `dataset-cache/` and OLMo-core builds the VSL caches on
  first run (adds a few hours to the first training job).
* We use the paper's §B.4 hyperparameters (LR 5e-3, wd 0.05), not the authors'
  `kas_config.json` example (LR 3e-4, wd 0.01, which is a 170M demo config).
  If a run diverges or loss spikes at this LR, that config is the fallback:
  `--lr 3e-4 --weight-decay 0.01`.
