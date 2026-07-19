"""Group 1 - "When the fact showed up in training".

All features are derived from the fact's appearance steps (the steps of its
`shared_chunks` -- chunks containing both subject and answer) and are computed
*relative to a checkpoint step*, since recency/appearance-count change as
training progresses.
"""
from __future__ import annotations

import numpy as np

import config
from src import schedule


def _appearances_up_to(appearance_steps, checkpoint_step):
    return [s for s in appearance_steps if s <= checkpoint_step]


def timeline_features(appearance_steps, checkpoint_step, total_steps=config.TOTAL_STEPS):
    """Temporal features for one fact evaluated at `checkpoint_step`.

    Parameters
    ----------
    appearance_steps : sorted list[int]
        Steps at which the fact's shared_chunks were presented (with multiplicity).
    checkpoint_step : int
        The checkpoint we are characterizing.
    """
    seen = _appearances_up_to(appearance_steps, checkpoint_step)
    f: dict[str, float] = {}
    f["n_appearances"] = len(seen)                    # cumulative count so far

    if not seen:
        # Fact not yet seen at this checkpoint -- leave temporal fields as NaN.
        f.update(dict(steps_since_last=np.nan, first_seen_step=np.nan,
                      last_seen_step=np.nan, span=np.nan,
                      gap_mean=np.nan, gap_std=np.nan, gap_cv=np.nan,
                      max_gap=np.nan, burstiness=np.nan,
                      lr_first=np.nan, lr_last=np.nan, lr_mean=np.nan,
                      first_epoch_frac=np.nan, last_epoch_frac=np.nan))
        return f

    first, last = seen[0], seen[-1]
    # --- recency: how long ago the model last saw it (STRONGEST feature) ------
    f["steps_since_last"] = checkpoint_step - last
    f["first_seen_step"] = first
    f["last_seen_step"] = last
    f["span"] = last - first                          # spread of appearances

    # --- spacing / clustering ------------------------------------------------
    gaps = np.diff(seen) if len(seen) > 1 else np.array([])
    f["gap_mean"] = float(gaps.mean()) if gaps.size else np.nan
    f["gap_std"] = float(gaps.std()) if gaps.size else np.nan
    f["gap_cv"] = (float(gaps.std() / gaps.mean())
                   if gaps.size and gaps.mean() > 0 else np.nan)
    f["max_gap"] = float(gaps.max()) if gaps.size else np.nan
    # Burstiness (Goh & Barabasi): -1 = perfectly spaced, +1 = bursty.
    if gaps.size:
        m, s = gaps.mean(), gaps.std()
        f["burstiness"] = float((s - m) / (s + m)) if (s + m) > 0 else 0.0
    else:
        f["burstiness"] = np.nan

    # --- learning rate at appearances ---------------------------------------
    lrs = [schedule.lr_at_step(s) for s in seen]
    f["lr_first"] = lrs[0]
    f["lr_last"] = lrs[-1]
    f["lr_mean"] = float(np.mean(lrs))

    # --- where in the epoch (early / middle / late as a 0..1 fraction) -------
    f["first_epoch_frac"] = first / total_steps
    f["last_epoch_frac"] = last / total_steps
    return f
