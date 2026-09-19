# frequency_forgetting/ — does forgetting get worse for facts the model sees more often?

**Question.** The LMEnt paper reports that frequently-seen facts are forgotten
more. Is that a property of the facts, or an artefact of counting: a fact that is
learned more often has more chances to be forgotten?

**Headline.** Forgetting *counts* rise with frequency, but the forgetting
*hazard* (forgetting events per opportunity) falls. Details and figures in the
notebook.

| file | role |
|---|---|
| `01_frequency_and_forgetting.ipynb` | **entry point** (CPU only): loads the 15-shot top-decile recall sweep, builds per-fact trajectories, and tests the count-vs-hazard question |
| `trajectory_analysis_rotemab_15shot_top10.ipynb` | companion: pivots the same sweep into the two CSVs below, plus learning/forgetting-frequency and competition explorations. Its last section reads a scratch `competition_df.parquet` that is not in the repo |
| `15_shot_top_10_trajectory.csv` | wide per-fact correctness matrix (id × checkpoint), written by the companion notebook |
| `15_shot_top_10_trajectory_with_fact_properites.csv` | per-fact `prop` (relation) and `num_shared_chunks` (exposure); also read by `neuron_forgetting/02_internal_paper_figures.ipynb` |

**Data read** (committed): `data/artifacts/recall/original_corpus/LMEnt_HF_models/15shot_top10pct_trajectory_1B-6E/internal/`
(`01`) and `data/artifacts/recall/original_corpus/LMEnt_HF_models/15shot_top10pct_trajectory_1B-6E/recall_only/` (the companion).

**Regenerate the sweep** (GPU): `feature_extraction/eval_checkpoints.py --num-shots 15
--top-shared-chunks-pct 10 --steps ...`; the exact command is in the notebook's first
markdown cell. Decoding is not bit-reproducible across reruns, so the committed
parquets are the source of the reported numbers.

The first code cell of each notebook locates the repo root, so they run from any
working directory.
