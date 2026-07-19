"""Training schedule: chunk -> steps mapping and the learning-rate curve.

`batch_indices.npy` maps each optimizer step to the chunk IDs seen in that
batch.  We invert it once into ``chunk_id -> sorted list of steps`` and cache
it, because every temporal feature needs per-fact appearance steps and scanning
the whole array per fact would be O(steps * batch) each time.

VERIFY: the on-disk shape/dtype of batch_indices.  We assume it is an
array-of-arrays (object array) or a 2-D int array [num_steps, batch_size].
"""
from __future__ import annotations

import math
import pickle
from collections import defaultdict

import numpy as np

import config


# --------------------------------------------------------------------------- #
# chunk_id -> steps
# --------------------------------------------------------------------------- #
def build_chunk_to_steps(batch_indices_path=None, cache_path=None) -> dict[int, list[int]]:
    """Invert batch_indices into {chunk_id: [steps...]} and cache to disk."""
    batch_indices_path = batch_indices_path or config.BATCH_INDICES_PATH
    cache_path = cache_path or config.CHUNK_INDEX_PATH

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    batch_indices = np.load(batch_indices_path, allow_pickle=True)
    chunk_to_steps: dict[int, list[int]] = defaultdict(list)
    for step, chunk_ids in enumerate(batch_indices):
        for cid in np.asarray(chunk_ids).ravel():
            chunk_to_steps[int(cid)].append(step)

    chunk_to_steps = {k: sorted(v) for k, v in chunk_to_steps.items()}
    with open(cache_path, "wb") as f:
        pickle.dump(chunk_to_steps, f)
    return chunk_to_steps


def steps_for_chunks(chunk_ids, chunk_to_steps: dict[int, list[int]]) -> list[int]:
    """All (step) appearances for a collection of chunk IDs, sorted, with
    multiplicity (a chunk can recur across batches)."""
    out: list[int] = []
    for cid in chunk_ids:
        out.extend(chunk_to_steps.get(int(cid), ()))
    return sorted(out)


# --------------------------------------------------------------------------- #
# Learning rate at a given step
# --------------------------------------------------------------------------- #
def lr_at_step(step: int,
               peak: float = config.LR_PEAK,
               warmup: int = config.LR_WARMUP_STEPS,
               total: int = config.TOTAL_STEPS,
               lr_min: float = config.LR_MIN) -> float:
    """Linear warmup then cosine decay to `lr_min`.

    VERIFY the schedule family and constants against LMEnt's training config;
    only the *shape* matters for the feature, but get the family right.
    """
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (peak - lr_min) * cosine
