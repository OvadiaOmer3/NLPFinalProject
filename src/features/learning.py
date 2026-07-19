"""Group 4 - "How the fact was learned" (from the acquisition timeline).

These are derived from the per-checkpoint recall trajectory produced by
``src.recall``.  IMPORTANT leakage note (from the proposal review): quantities
like the number of forgetting events or oscillation count are essentially part
of the *target* you are predicting, not a cause.  They are computed here for EDA
only and returned under a separate ``leaky_`` prefix so they are trivial to drop
before fitting the prediction model.
"""
from __future__ import annotations

import numpy as np

import config


def _first_appearance(appearance_steps):
    return appearance_steps[0] if len(appearance_steps) else np.nan


def learning_features(traj, appearance_steps,
                      checkpoint_steps=config.CHECKPOINT_STEPS):
    """
    Parameters
    ----------
    traj : list of dict, one per checkpoint (ascending step) with keys
        {"step", "correct" (bool), "gold_prob" (float)}.
    appearance_steps : sorted list[int]
        Training steps where the fact's shared_chunks appeared.
    """
    traj = sorted(traj, key=lambda d: d["step"])
    steps = np.array([d["step"] for d in traj])
    correct = np.array([bool(d["correct"]) for d in traj])
    gold = np.array([d["gold_prob"] for d in traj], dtype=float)

    f: dict[str, float] = {}
    first_app = _first_appearance(appearance_steps)

    # --- first-learned step & confidence there --------------------------------
    if correct.any():
        first_idx = int(np.argmax(correct))            # first True
        first_learned = int(steps[first_idx])
        f["first_learned_step"] = first_learned
        f["confidence_at_first_known"] = float(gold[first_idx])
        # how fast: steps between first exposure and first correct
        f["steps_to_learn"] = (first_learned - first_app
                               if not np.isnan(first_app) else np.nan)
    else:
        f["first_learned_step"] = np.nan
        f["confidence_at_first_known"] = np.nan
        f["steps_to_learn"] = np.nan

    # peak confidence ever reached (still a legitimate predictor of robustness)
    f["max_gold_prob"] = float(np.nanmax(gold)) if np.isfinite(gold).any() else np.nan

    # --- leaky / target-like signals: EDA ONLY, drop before modeling ----------
    flips = np.diff(correct.astype(int))
    f["leaky_forgetting_events"] = int((flips == -1).sum())     # 1 -> 0 transitions
    f["leaky_relearning_events"] = int((flips == 1).sum())      # 0 -> 1 transitions
    f["leaky_oscillation_count"] = int((flips != 0).sum())
    f["leaky_ever_correct"] = bool(correct.any())
    f["leaky_correct_final"] = bool(correct[-1]) if len(correct) else False
    return f


def retention_target(traj):
    """The prediction target: did a fact that was learned survive to the end?

    Returns 1 (retained), 0 (forgotten), or None (never learned -> excluded).
    A fact is "learned" if it is ever correct; "retained" if also correct at the
    final checkpoint.
    """
    traj = sorted(traj, key=lambda d: d["step"])
    correct = [bool(d["correct"]) for d in traj]
    if not any(correct):
        return None                     # never learned -> not part of the study
    return int(correct[-1])
