"""Generate the few-shot prompt file the eval reads (see ``src/prompt_set.py``).

Build the prompts once, then evaluate any number of checkpoints or models against
that file -- the prompts, the fact set and its order are then constant by
construction instead of by matching flags.

Usage:
    python feature_extraction/build_prompts.py --num-shots 15
    python feature_extraction/build_prompts.py --num-shots 5 \
        --fewshot-mode only-same-prop            # -> prompts_5shot_onlysameprop.parquet
    python feature_extraction/build_prompts.py --num-shots 0        # bare cloze
    python feature_extraction/build_prompts.py --num-shots 15 --min-shared-chunks 1
    python feature_extraction/build_prompts.py --inspect data/prompts/prompts_15shot.parquet

Then:
    python feature_extraction/eval_checkpoints.py --prompts data/prompts/prompts_15shot.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prompt_set import main

if __name__ == "__main__":
    main()
