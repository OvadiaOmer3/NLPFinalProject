# neuron_forgetting/ — what forgetting looks like inside the model

**Question.** When a fact flips from correct to wrong between checkpoints, what
changes inside the network: where in the stack the answer stops being readable,
and which MLP neurons stop firing?

**Headline.** Forgetting is a mid-stack failure (the logit-lens gap opens around
layer boundary 8), neuron churn precedes the flip, and it adds predictive signal
over the output-level scalars (AUC 0.713 vs 0.694). Details in the notebook.

| file | role |
|---|---|
| `02_internal_paper_figures.ipynb` | **entry point**: all internal-state figures (logit lens, MLP activity, neuron overlap / churn) |
| `features_02_internal_paper_figures.md` | per-figure dictionary of the features the notebook uses |
| `appendix_internal_features.md` | the extraction protocol behind `src/features/internal.py` (one teacher-forced pass per fact and checkpoint; hidden-state and top-k neuron dumps) |

Shared code this uses: `src/features/internal.py` (extractor, `NeuronDump`), run
through `feature_extraction/eval_checkpoints.py`; tests in
`tests/test_neuron_dump.py` and `tests/test_fewshot_internals.py`.

**Data read:** `data/artifacts/recall/rotemab_15shot_top10_internal/` (committed),
`frequency_forgetting/15_shot_top_10_trajectory_with_fact_properites.csv` (committed),
and `data/artifacts/expermints/15shot_trajectory_top10_internal_with_neurons/`
(gitignored, ~5 GB of neuron dumps). Regenerate the latter with
`feature_extraction/eval_checkpoints.py --top-k 128 ...` (command in the notebook's
first markdown cell) before running the neuron-overlap cells; the other figures run
from the committed parquets.

The first code cell locates the repo root, so the notebook runs from any working
directory.
