# Token interference and forgetting

**Trajectory: LMEnt-1B-6E, 15-shot `drop-same-prop`, top-10% by co-occurrence.**
Run 2026-08-08.

## Conclusion

**Subject-side token interference predicts forgetting in the hypothesised direction,
but the evidence stops short of establishing it as a token-identity effect.**

Three findings, in descending order of confidence:

1. **Subject strain is a significant predictor, and the largest well-identified
   effect in the model** — more entities sharing the subject's tokens, more
   forgetting. +0.189 log-odds per log-unit (p=0.0016), which is **+0.293 per
   standard deviation, an odds ratio of 1.34**; it adds chi2(2)=19.3 (p=6.6e-05)
   over a baseline that already holds exposure, popularity, name length, and
   relation and window fixed effects.
2. **Name length is not the confound on this subset** (chi2(2)=3.7, p=0.16). This
   matters because on a broader fact set it is: strain rises steeply with how many
   tokens a name has, and there name length dominates. Restricting to the
   high-co-occurrence decile removes that.
3. **But a permutation test does not clear significance** (p=0.073). Rebuilding the
   token index with tokens randomly reassigned across entities — preserving each
   entity's token *count* and the corpus-wide token frequencies, destroying only
   which token belongs to whom — leaves 10 of 150 scrambled indices scoring at
   least as high as the real one. So we cannot rule out that the effect comes from
   structural properties the shuffle preserves rather than from actual token
   collisions.

The object side shows nothing (−0.025, p=0.72), and the effect is concentrated in
two of five relations rather than broad-based.

**Read this as: the hypothesis survived its sharpest test but did not pass it.**

---

## What was done

**Outcome.** 15-shot `drop-same-prop` recall sweep over 13 checkpoints
(50,000 → 658,032; no step 0). Demonstrations never share the query's relation, so
in-context answer-type leakage is designed out. 1,452 facts raw → **1,310** after
dropping 142 `excluded` (the answer or an alias occurs inside the question, so they
are answerable by copying the prompt). Accuracy 0.147 → 0.415.

**Fact subset.** `num_shared_chunks >= 42` — the **top 10.2%** of popqa-kas by
subject/answer co-occurrence.

This is a high-frequency slice, **not** a fixed-frequency one: within it
`num_shared_chunks` still runs median 499, p90 7,133, max 101,081. That matters for
one specific reading of the results. `l_evidence_static` comes out null here
(p=0.59), and the cheap explanation would be "restricted range — there is no
frequency variation left to predict with". There is: ~14x spread from median to p90.
So exposure being null is a fact about this subset, not an artifact of having
squeezed the variance out of it. What the restriction *does* buy is that the
frequency confound is much weaker than on the full fact set, which is why name
length stops mattering here.

**Panel.** One row per (fact, checkpoint window) where the fact was **correct at the
window's start** — a discrete-time hazard design; a fact cannot be forgotten if it
was never known. 12 windows, **4,163 rows / 1,006 forgetting events / 713 facts**.
After dropping relations with <100 rows or <20 events: **3,932 / 878 / 623 facts
over 5 relations** (`capital` 2,401, `capital of` 721, `religion` 335, `sport` 250,
`country` 225).

**Features.** Competitors are counted as **entities** (Wikidata QIDs), not surface
strings, so two distinct entities written "Buffalo" count twice and an entity never
competes with its own aliases. Canonical surface forms only — Wikidata alias lists
are a knowledge-base artifact, not evidence a token appeared in training.

### Every column, in words

| column | meaning |
|---|---|
| `forgot` | **outcome.** 1 if the fact was correct at the window's start and wrong at its end. |
| `subj_strain`, `obj_strain` | **the hypothesis.** How many *other entities* share ≥1 token with this fact's subject / object. Computed as the **size of the union** of rival sets across the entity's tokens — not a sum, not a mean — excluding the fact's own other endpoint. |
| `subj_strain_w`, `obj_strain_w` | the same, restricted to entities that appeared in *this window*. |
| `evidence_static` | **exposure.** Chunks where the subject co-occurs with either the true answer or a rival answer of the same relation — the relevant material the model actually saw. |
| `evidence_w` | exposure restricted to this window. |
| `s_pop`, `o_pop` | Wikipedia sitelink counts for subject / object. Real-world popularity, not exposure. |
| `s_ntok`, `o_ntok` | **name length.** Number of tokens in the subject / object name, after the frequency cutoff drops `the`/`of`. |
| `s_own_rare`, `o_own_rare` | **F1, ownership.** This fact's evidence as a *share*: `num_shared_chunks ÷ |all chunks carrying the entity's rarest token|`. Low = the fact's signal is a small slice of everything bearing those tokens. |
| `s_rivals_rare`, `o_rivals_rare` | **F1, rivals.** How many other entities carry that rarest token. |
| `ratio_static`, `ratio_w` | **F2, alt-answer.** `(shared+1) ÷ (shared+distract+2)`, where `distract` counts chunks pairing the subject with a *different* valid answer for the same relation. High = the evidence the model saw was mostly the right answer. |

All continuous predictors enter as `log(1+x)` (prefix `l_`), so a unit is roughly an
e-fold change in the raw quantity.

### The models

Logistic regression on `forgot`, standard errors clustered by fact. Every model
contains `C(prop)` and `C(window)` — a separate intercept per relation and per
window — so comparisons are *within* relation and *within* window.

```
BASE      = forgot ~ l_evidence_w + l_evidence_static + l_s_pop + l_o_pop
                     + C(prop) + C(window)
NAMELEN   =        + l_s_ntok + l_o_ntok
STRAIN    =        + l_subj_strain + l_obj_strain          (or the _w variants)
F1        =        + l_s_own_rare + l_o_own_rare + l_s_rivals_rare + l_o_rivals_rare
F2        =        + ratio_static                          (or ratio_w)
```

So the table row **`+ strain | name length`** is literally
`BASE + NAMELEN + STRAIN` — the pipe reads "strain, *given* name length is already
controlled". Nothing is dropped; terms are only ever added.

**Why the same model shows two different χ² values.** A likelihood-ratio test needs a
reference model, and the doc uses two:

| reference | asks |
|---|---|
| `BASE` (exposure only) | does this add anything over frequency and popularity? |
| `BASE + NAMELEN` | does this add anything *beyond name length* — the confound that explains the same association on broader fact sets? |

`BASE + NAMELEN + STRAIN` scores **χ²(4)=22.9** against `BASE` (4 added terms: 2 name
length + 2 strain) and **χ²(2)=19.3** against `BASE + NAMELEN` (2 added terms: strain
only). Same fitted model, different question. **The second is the one that matters**,
because strain and name length are correlated, so a comparison that credits strain
with name length's work is not evidence about interference.

Name length enters as post-cutoff token count only. `charlen` (raw string length)
measures the same construct at spearman 0.88 and splitting collinear credit between
them makes both uninterpretable; it is kept in the panel as a robustness check.

---

## Results

### Model comparison

```
model (all include C(prop) + C(window))   pseudoR2   vs BASE          vs BASE+NAMELEN
BASE  exposure + popularity                 0.1567   --               --
BASE + NAMELEN                              0.1575   chi2(2)=  3.7    --
                                                     p=0.16
BASE + STRAIN                               0.1592   chi2(2)= 10.4    (not nested)
                                                     p=5.5e-03
BASE + NAMELEN + STRAIN                     0.1622   chi2(4)= 22.9    chi2(2)= 19.3
                                                     p=1.3e-04        p=6.6e-05   <-- headline
BASE + NAMELEN + F1                         0.1643   chi2(6)= 31.8    chi2(4)= 28.1
                                                     p=1.8e-05        p=1.2e-05
BASE + NAMELEN + F2                         0.1576   chi2(3)=  3.9    chi2(1)=  0.2
                                                     p=0.28           p=0.66
```

Read the **right-hand column** for anything about interference; the middle column
credits each feature with name length's contribution as well.

The F1 row is optimizer-sensitive: its four correlated terms leave the likelihood
nearly flat, and BFGS has landed on chi2 = 28.1 and 28.7 in different sessions.
Reproducible within a session, so do not quote it to three significant figures. The
strain rows are stable.

### Coefficients (static strain, name length controlled, clustered by fact)

Predictors are `log(1+x)` but **not standardised**, so raw coefficients are per
log-unit of their own variable and are **not comparable across rows**. The
`coef x SD` column rescales each to "change in log-odds per 1 SD of that predictor",
which is the column to read when ranking effects. Sorted by that.

```
                    coef   Std.Err.   SD(x)   coef x SD   OR per SD    P>|z|
l_evidence_static -0.4525   0.8305    1.828    -0.827       0.44      0.5858
l_evidence_w      +0.2190   0.8261    1.826    +0.400       1.49      0.7906
l_subj_strain     +0.1894   0.0599    1.547    +0.293       1.34      0.0016  <- hypothesis
l_o_pop           -0.1945   0.0627    1.327    -0.258       0.77      0.0019
l_s_ntok          -0.8264   0.3401    0.291    -0.241       0.79      0.0151
l_s_pop           -0.0908   0.0810    1.459    -0.132       0.88      0.2623
l_o_ntok          +0.4784   0.3931    0.249    +0.119       1.13      0.2236
l_obj_strain      -0.0247   0.0680    1.498    -0.037       0.96      0.7164
```

Two things this reordering changes:

**Subject strain is the largest well-identified effect in the model**, not a small
one. On raw coefficients name length looks ~4x bigger (−0.83 vs +0.19); that is an
artifact of spread. `s_ntok` is 1–6 tokens with median 1, so its log SD is 0.29,
while `subj_strain` runs 0–422 with log SD 1.55. Per SD the ordering reverses:
+0.293 for strain against −0.241 for subject name length. A standard-deviation move
in subject strain multiplies the odds of forgetting by **1.34**.

**The two exposure terms are not interpretable** and should not be read as effects.
They carry the largest `coef x SD` values but also standard errors ~4x their
coefficients (p=0.59, p=0.79), because `evidence_w` is a subset of
`evidence_static` — near-perfectly collinear, so the fit splits a large amount of
shared variance arbitrarily between them. They are doing their job as controls;
their individual coefficients are noise.

Raw forgetting rate by subject-strain quartile — no model, no controls:

```
Q1 low   0.207      Q3       0.229
Q2       0.175      Q4 high  0.283
```

### Permutation test (150 draws, name length in the baseline)

```
real token index      chi2 = 19.3
null distribution     min 0.0 | median 5.9 | max 41.7 | mean 7.7 (sd 7.6)
permutations >= real  10 of 150
empirical p           0.073
```

### Within-relation breakdown

```
relation      rows  events   l_subj_strain
all (5)       3932     878   +0.189 (se 0.060, p=0.002)
capital       2401     410   +0.060 (se 0.093, p=0.520)
capital of     721     123   +0.660 (se 0.231, p=0.004)
religion       335     102   +0.400 (se 0.123, p=0.001)
sport          250     109   +0.088 (se 0.194, p=0.651)
country        225     134   +0.098 (se 0.236, p=0.679)
```

### Windowed features

```
windowed on top of static   chi2(2)= 2.2   p=0.33
static on top of windowed   chi2(2)= 3.0   p=0.23
```

---

## How the results support the conclusion

**For the hypothesis.** The subject-strain coefficient is positive and significant
in exactly the predicted direction, it survives an exposure baseline built
specifically so that frequency cannot explain it, the raw quartiles rise without any
model, and name length — the confound that explains the same association on broader
fact sets — is not significant here.

**Against treating it as established.** The permutation test is the one that
distinguishes "which tokens collide" from "some structural property correlated with
having many tokens". At p=0.073 it does not separate them. And the pooled effect is
carried by `capital of` (+0.66) and `religion` (+0.40) while being null in `capital`
(+0.06), which is 61% of the rows, and in `sport` and `country`. A mechanism that
operated on token collision should not be that relation-specific.

**On the object side**, nothing: −0.025, p=0.72. If the mechanism were competition at
the decoding step — many entities competing for the answer's tokens — the object side
is where it should appear most strongly. It does not.

**The windowed form is uninformative here**, as on every panel tried: about 90% of
the 19,384-entity universe is active in every 50,000-step window, so "which
competitors appeared since the last checkpoint" is effectively "all of them".
Windowed strain is a near-copy of static strain, and adds nothing to it.

---

## What would settle it

1. **Finer checkpoints.** The median object entity occupies 305 of ~10.5M chunks, so
   a window must cover well under 1/305 of the corpus before absence is common —
   roughly **1–2K steps, 25–50x finer than the current grid**. The 6-epoch model
   ships ~110 checkpoints per epoch, so this needs ~110 evaluations rather than 13.
   Until then the *temporal* form of the hypothesis is untestable, and the temporal
   form is the one that could show interference happening rather than merely
   correlating.
2. **A corpus-wide competitor universe.** Competitors are currently the 19,384
   entities in popqa-kas, against millions in the corpus, so strain is attenuated —
   and attenuated more for rare entities. The mention annotations are 32 GB of
   streamable `.csv.gz` (`dataset-tokenized/part-*.csv.gz`), not the 127 GB
   Elasticsearch tarball.
3. **More relations at this exposure level.** Five relations, two carrying the
   effect, is too thin to tell a mechanism from a relation-specific quirk.

---

## Caveats

1. **713 facts / 878 events.** Small. Null results here are weak evidence; the
   positive one needs replication before it is built on.
2. **Relation mix is extreme** — `capital` and `capital of` are 79% of rows, and
   `occupation` and `place of birth` drop out entirely. Relation fixed effects make
   comparisons within-relation, but they cannot make the sample representative.
3. **Exposure control has little to bite on.** The subset is the top decile by
   co-occurrence, so the frequency confound is largely designed out — but so is the
   variation the exposure terms would use, which is why `l_evidence_static` is null.
4. **Multiplicity.** Several feature families were tested against one panel. The
   nominal p=0.0016 on subject strain is not adjusted for that.
5. **`correct` is a substring match** — any alias longer than three characters
   anywhere in a 15-token greedy generation. Some of the 1,006 events are scoring
   flips rather than knowledge flips; an exact-match arm would bound this.
6. **Tokens are whitespace words.** BPE is implemented (`--bpe`) but not run; the
   frequency cutoff would need re-deriving for subword pieces.

---

## Reproducing

```bash
python feature_extraction/build_strain_panel.py \
    --recall-dir data/artifacts/expermints/15shot_trajectory_top10_internal_with_hidden/recall \
    --out data/artifacts/features/strain_panel_15shot_top10.parquet

python analysis/strain_models.py \
    --panel data/artifacts/features/strain_panel_15shot_top10.parquet

python analysis/null_control_distribution.py \
    --panel data/artifacts/features/strain_panel_15shot_top10.parquet \
    --n 150 --control-namelen \
    --out data/artifacts/features/null_control_15shot_top10_namelen.json
```

Feature code `src/features/strain.py`; 15 regression tests in
`tests/test_strain_features.py`, covering the three mistakes this metric already
made — same-string/different-QID must count twice, an entity must not compete with
itself, and a subject must not be its own distractor.

**Run the permutation test with `--control-namelen`.** Without it the baseline is
exposure-only, and there the null distribution centres far from zero because
scrambled strain remains a proxy for name length — so both the real and the
scrambled feature beat that baseline for the same uninteresting reason, and the test
answers the wrong question.
