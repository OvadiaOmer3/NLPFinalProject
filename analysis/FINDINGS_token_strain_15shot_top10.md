# Token interference — replication on the 15-shot / top-10% trajectory

**Companion to [`FINDINGS_token_strain.md`](FINDINGS_token_strain.md), which covers
the 5-shot / full-set trajectory. Neither supersedes the other; they are two
independent trajectories and the outputs are kept separate.**

Run date 2026-08-07.

```bash
python feature_extraction/build_strain_panel.py \
    --recall-dir data/artifacts/expermints/15shot_trajectory_top10_internal_with_hidden/recall \
    --out data/artifacts/features/strain_panel_15shot_top10.parquet
python analysis/strain_models.py --panel data/artifacts/features/strain_panel_15shot_top10.parquet
python analysis/null_control_distribution.py \
    --panel data/artifacts/features/strain_panel_15shot_top10.parquet --n 30 \
    --out data/artifacts/features/null_control_15shot_top10.json
```

## Which artifact belongs to which trajectory

| | 5-shot / full | 15-shot / top-10% |
|---|---|---|
| source | `5_shot_trajectory_with_fact_properites.csv` | `data/artifacts/expermints/15shot_trajectory_top10_internal_with_hidden/recall/` |
| few-shot | 5-shot, `only-same-prop` | 15-shot, `drop-same-prop` |
| facts | 14,109 | 1,452 raw → **1,310** after dropping 142 `excluded` |
| fact subset | all | `num_shared_chunks >= 42` (**top 10.2%** of popqa) |
| checkpoints | 14 (0 → 658,032) | 13 (**50,000** → 658,032; no step 0) |
| accuracy | 0.006 → 0.044 | 0.147 → 0.415 |
| panel | `strain_panel.parquet` | `strain_panel_15shot_top10.parquet` |
| null control | `null_control.json` | `null_control_15shot_top10.json` |
| findings | `FINDINGS_token_strain.md` | this file |

Panel: 12 windows, **4,163 rows / 1,006 forgetting events / 713 facts**; after the
relation filter, 3,932 / 878. Facts overlapping the 5-shot set: 1,411 of 1,452.

Two differences worth holding in mind when comparing. This subset is the top decile
by co-occurrence, so **exposure barely varies** — the frequency confound is largely
designed out, but the exposure control also has little to bite on. And relations are
heavily skewed: `capital` (442) and `capital of` (231) are 46% of the sample, while
`occupation` (4) and `place of birth` (9) drop out entirely.

## Result: the same conclusion, reached independently

```
model                               pseudoR2         LR vs exposure-only
exposure only                         0.1567                          --
+ NAME LENGTH                         0.1613  chi2(4)=   19.5  p=6.2e-04
+ strain STATIC (no len ctrl)         0.1592  chi2(2)=   10.4  p=5.5e-03
+ everything | name length            0.1730  chi2(11)=  68.0  p=2.9e-10

Incremental over the NAME-LENGTH baseline:
  + strain STATIC | name length      chi2(2)=  17.7  p=1.5e-04
  + strain WINDOWED | name length    chi2(2)=  16.3  p=2.8e-04
  + F1 ownership/rivals | len        chi2(4)=  36.5  p=2.3e-07
  + F2 ratio WINDOWED | len          chi2(1)=   0.0  p=0.94
```

**1. The null control fails again.** 30 permutations, tokens reassigned across
entities with per-entity token counts and global frequencies preserved:

```
                    real chi2    null min / median / max      perms >= real    p
5-shot full            50.6        17.4 / 48.5 / 116.2         14 of 30      0.484
15-shot top-10%        10.4         0.1 /  7.9 /  30.7         12 of 30      0.419
```

Both land at the median of their own null distribution. Two trajectories differing
in shot count, few-shot mode, fact subset and accuracy regime give the same verdict:
**the feature does not depend on which tokens an entity has.**

**2. Neither strain term survives in the combined model.** Unlike the 5-shot run —
where object-side strain was significant until name length was controlled — here it
is not significant even before that:

```
                 Coef.   Std.Err.  P>|z|
l_subj_strain_w +0.1349   0.0847   0.111
l_obj_strain_w  -0.1046   0.1215   0.389
```

And the raw quartile table is **non-monotone** (0.240 / 0.159 / 0.202 / 0.292),
where the 5-shot run's was cleanly increasing. That monotonicity was the
name-length confound; on a subset where name length matters less (chi2 19.5 here
vs 250.1 there), it disappears.

**3. Windowed still adds nothing over static** (chi2(2)=4.5, p=0.11; static over
windowed chi2(2)=5.8, p=0.055) — same saturation as before, ~17,500 of 19,384
entities active in every window.

## What differs from the 5-shot run

**Name length is far weaker here**: chi2(4)=19.5 vs 250.1. Expected — this is a
narrow high-frequency slice, so there is less spread in the things name length
proxies for.

**Subject name length splits in two**, which the 5-shot run did not show:

```
l_s_ntok      -1.3702  p=0.006   more subject TOKENS  => less forgetting
l_s_charlen   +0.6470  p=0.010   longer subject STRING => more forgetting
l_o_charlen   +0.6307  p=0.021   longer object STRING  => more forgetting
```

`l_s_ntok` reproduces the 5-shot estimate almost exactly (−1.37 vs −1.33), so the
"more specific key is protective" reading survives replication. But `l_s_charlen`
now points the other way at similar significance, which the 5-shot run did not
show (−0.09, p=0.47). Token count and character count are close proxies, so
opposite signs at n=713 facts is a collinearity warning, not two findings. Do not
build on the character-length term without a design that separates them.

**F1 is the strongest family here** (chi2(4)=36.5 over name length), driven by the
subject side: `l_s_own_rare` +0.198 (p=0.0015), `l_s_rivals_rare` +0.247 (p=0.022).
It was also the strongest in the 5-shot run. Since F1 survived neither
trajectory's null control as an *interference* measure, the most likely reading is
that it too is carrying entity-structure information rather than token identity —
worth its own permutation test with the ownership features held out.

## Caveats specific to this run

1. **No step 0**, so the panel starts at 50,000 and has 12 windows, not 13. Windows
   are not aligned with the 5-shot panel's; do not pool the two.
2. **142 `excluded` facts dropped** (answer occurs in the question) — these are
   answerable by copying from the prompt. The 5-shot CSV kept them. Re-run with
   `--keep-excluded` to match the older treatment.
3. **713 facts / 878 events** after filtering is ~4x smaller than the 5-shot panel,
   so null results here are weaker evidence than the 5-shot ones. The positive
   result that replicates (`l_s_ntok`) is the more trustworthy signal.
4. Internal-state columns (`gold_prob`, `mlp_*`, `resolution_depth`, ...) are
   present in this sweep and **unused** by the strain panel. They are a real
   opportunity the 5-shot trajectory did not offer.
