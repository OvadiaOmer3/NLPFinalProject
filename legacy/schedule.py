"""Training schedule: chunk -> steps mapping and the learning-rate curve.

`batch_indices.npy` maps each optimizer step to the chunk IDs seen in that
batch.  We invert it once into ``chunk_id -> sorted list of steps`` and cache
it, because every temporal feature needs per-fact appearance steps and scanning
the whole array per fact would be O(steps * batch) each time.

On disk batch_indices is an object array of 109,672 steps x 16 chunk ids per
epoch (verified on download); a 2-D int array [num_steps, batch_size] also works.
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
               alpha_f: float = config.LR_ALPHA_F) -> float:
    """LMEnt's training LR: linear warmup then cosine decay to ``alpha_f * peak``.

    Mirrors OLMo-core ``olmo_core.optim.scheduler.CosWithWarmup`` (the class used
    by ``kas_config.json``): for ``t`` past warmup with horizon ``t_max``,
        lr = alpha_f*peak + (1 - alpha_f)*peak * 0.5*(1 + cos(pi * t/t_max)).
    Constants come from the real config (see config.py); confirm ``alpha_f`` for
    the 1B run at checkout.  If OLMo-core is installed, prefer its scheduler
    directly rather than this re-derivation.
    """
    if warmup > 0 and step < warmup:
        return peak * step / warmup
    t = step - warmup
    t_max = max(1, total - warmup)
    frac = min(1.0, max(0.0, t / t_max))
    cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
    return alpha_f * peak + (1.0 - alpha_f) * peak * cosine
