# Token interference and forgetting — result: hypothesis not supported

**Trajectory: 5-shot `only-same-prop`, all 14,109 facts, 14 checkpoints (0 →
658,032).** Artifacts: `strain_panel.parquet`, `null_control.json`.
Independently replicated on a 15-shot `drop-same-prop` top-10% trajectory —
see [`FINDINGS_token_strain_15shot_top10.md`](FINDINGS_token_strain_15shot_top10.md),
whose artifacts carry the `_15shot_top10` suffix. Neither run supersedes the other.

Run date 2026-08-02. Reproduce with:

```bash
python feature_extraction/build_strain_panel.py \
    --trajectory 5_shot_trajectory_with_fact_properites.csv \
    --out data/artifacts/features/strain_panel.parquet
python analysis/strain_models.py --panel data/artifacts/features/strain_panel.parquet
```

## Question

LMEnt §6.1 finds fact frequency correlates with both learning *and* forgetting and
says frequency "does not dictate learning and forgetting dynamics, and may be
confounded by other factors." Candidate confounder tested here: **token
interference** — a fact is hard to retain when its tokens are shared with many
other entities the model saw nearby in training.

## Design

Panel: one row per (fact, checkpoint-window) where the fact was **correct at the
window's start**, i.e. genuinely at risk. 13 windows over steps 0→658,032.
**11,328 rows, 3,167 forgetting events, 2,213 facts**; after dropping relations
with <100 rows or <20 events, 10,450 / 2,849 / 1,979 over 9 relations.
Logit with relation and window fixed effects, SEs clustered by fact. Every
feature is scored as an addition to an **exposure-only baseline**, never against
an empty model.

Competitors are counted as **entities (QIDs)**, using canonical surface forms
only — not Wikidata aliases, which are a knowledge-base artifact rather than
evidence a token ever appeared in training.

## Result

```
model                               pseudoR2         LR vs exposure-only
exposure only                         0.1368                          --
+ NAME LENGTH                         0.1572  chi2(4)=  250.1  p=6.1e-53
+ strain STATIC (no len ctrl)         0.1398  chi2(2)=   37.5  p=7.3e-09
+ everything | name length            0.1632  chi2(11)= 323.3  p=1.1e-62
```

**Name length is ~7x stronger than token strain, and it explains it.**

Two checks killed the interference reading:

1. **The null control failed — as a full permutation test, not one draw.**
   The token index is rebuilt with tokens randomly reassigned across entities:
   each entity keeps its token *count*, the corpus-wide token frequencies are
   preserved, and only *which* token belongs to whom is destroyed. Measured on
   the real data, the shuffle preserves per-entity token count for 19,371 of
   19,384 entities (99.93%) while leaving just **53 of 37,574 entity–token pairs
   (0.14%)** intact.

   Over **30 permutations** (`analysis/null_control_distribution.py`, seeds 0-29,
   reproducible):

   ```
   real token index         chi2 = 50.6
   null distribution        min 17.4 | median 48.5 | max 116.2 | mean 53.4 (sd 23.5)
   permutations >= real     14 of 30
   empirical p              0.484
   ```

   The real index lands at the **median** of its own null distribution. Whatever
   "strain" measures, it is not *which* tokens an entity has — it is something the
   shuffle preserved. Spearman(n_tokens, strain) is 0.61 real and still 0.41
   permuted, which identifies that something as name length.

   (An earlier single-draw version of this check reported one scrambled index at
   chi2=85.1 and read it as "the null scores *higher*". That draw was simply in the
   upper half of a wide null distribution; the distribution above supersedes it. The
   permutation is now seeded deterministically — token sets are sorted before
   shuffling, since Python randomises string hashes per process and the earlier
   version was not reproducible across runs.)

2. **Controlling name length reverses the sign.** Strain rises steeply with how
   many tokens a name has (mean strain 4.8 / 28 / 72 / 109 / 127 for 1..5-token
   names; spearman 0.61) — more tokens simply means more chances to collide.
   With name length in the model, object-side strain goes from **+0.098
   (p=0.001) to −0.030 (p=0.66)**.

The raw quartile table still looks monotone (forgetting 0.236 / 0.238 / 0.276 /
0.341 by object-strain quartile), but that is the name-length confound showing
through: high-strain objects are long-named objects.

## What does predict forgetting

```
                    Coef.   Std.Err.  P>|z|
l_s_ntok           -1.3322   0.2332  <0.0001   <- longer SUBJECT name => much LESS forgetting
l_o_charlen        +0.4231   0.1623   0.0091   <- longer OBJECT string => MORE forgetting
l_o_pop            -0.1150   0.0413   0.0054
l_evidence_static  -0.2559   0.0950   0.0070
```

Both are plausible without any interference story:

* A longer **object** is harder to emit exactly. `is_correct` is a substring
  match over a 15-token greedy generation, so a 4-token answer has more ways to
  come out wrong than a 1-token answer. Partly a scorer artifact — worth
  re-checking under exact match.
* A longer **subject** is a more specific retrieval key. "Joshua Patrick Allen"
  disambiguates where "Allen" does not. This is adjacent to the original
  hypothesis — it is about the *specificity of the key* rather than about token
  collision — and is the thread worth pulling next.

## The windowed feature is degenerate at this checkpoint spacing

```
windowed on top of static   chi2(2)= 1.3  p=0.51
static on top of windowed   chi2(2)= 0.7  p=0.70
```

Diagnostic: **corr(static, windowed) = 0.997**, median windowed/static ratio
0.95, median within-fact coefficient of variation across windows **0.035**. Cause:
**~17,500 of 19,384 entities (90.3%) are active in every 50K-step window.** At
this spacing "which competitors appeared since the last checkpoint" is
approximately "all of them".

Sizing the fix: the median object entity occupies 305 of ~10.5M chunks, so a
window must cover well under 1/305 of the corpus for absence to be common —
roughly **1–2K steps, i.e. 25–50x finer than the current grid**. The 6E model
ships ~110 checkpoints per epoch, so this is available, but it means evaluating
~110 checkpoints instead of 14. **Until that exists, no windowed interference
claim is testable on this data** — that, rather than the null result above, is
the real blocker on the original hypothesis.

## Caveats

1. Competitor universe is PopQA-only (19,384 entities vs. millions in the
   corpus), so strain is attenuated, and attenuated more for rare entities.
2. Tokens are whitespace words. BPE (`--bpe`) is implemented but not run here;
   the df cutoff should be re-derived for subword pieces.
3. Outcome is `only-same-prop` 5-shot, where demonstrations share the query's
   relation and so are maximally token-overlapping with the target by
   construction. A `drop-same-prop` trajectory is expected and will be rerun
   through the same panel (the features do not depend on it).
4. `correct` is a substring match; some of the 3,167 events are scorer noise.
