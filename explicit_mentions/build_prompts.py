"""Generate the few-shot prompt file eval_explicit.py reads (see emlib/prompt_set.py).

Build the prompts once, then evaluate the baseline model and every explicit-corpus
model against that same file: the prompts, the fact set and its order are then
constant by construction, so a difference in `correct` is a difference between
models rather than between prompt renderings.

Usage:
    python build_prompts.py --num-shots 5           # -> prompts_5shot_onlysameprop.parquet
    python build_prompts.py --num-shots 5 --fewshot-mode drop-same-prop
    python build_prompts.py --num-shots 0           # bare cloze
    python build_prompts.py --inspect <EXPLICIT_ROOT>/prompts/prompts_5shot_onlysameprop.parquet

Then, for every model:
    python eval_explicit.py --model dhgottesman/LMEnt-170M-1E --steps 109672 \
        --run-tag baseline_170M_1E --prompts <...>/prompts_5shot_onlysameprop.parquet
    python eval_explicit.py --model <...>/hf/explicit_170M_1E --steps 109672 \
        --run-tag explicit_170M_1E --prompts <...>/prompts_5shot_onlysameprop.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emlib.prompt_set import main

if __name__ == "__main__":
    main()
