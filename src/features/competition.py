"""Group 2 - "Competition from other facts" (interference).

Every time the subject appears in a chunk that does NOT state the fact, the
gradient update pulls the subject's representation in some other direction.
Same on the answer side.  The proposal only fixes the *combined* subject+answer
count, so we record subject and answer counts separately, and -- crucially --
the interference that happens *after* the fact was first learned.
"""
from __future__ import annotations

import numpy as np

from src import schedule


def competition_features(row, chunk_to_steps, checkpoint_step,
                         first_learned_step=None):
    """Interference features for one fact at `checkpoint_step`.

    Parameters
    ----------
    row : mapping with subject_chunks, answer_chunks, shared_chunks (+ counts)
    chunk_to_steps : dict[int, list[int]] from schedule.build_chunk_to_steps
    first_learned_step : int | None
        If known (from the recall timeline), we also count interference that
        occurred strictly after the fact was learned -- the updates that could
        overwrite it.
    """
    subject_chunks = set(int(c) for c in row["subject_chunks"])
    answer_chunks = set(int(c) for c in row["answer_chunks"])
    shared_chunks = set(int(c) for c in row["shared_chunks"])

    # "subject-only" = subject appears but the fact is not stated there.
    subject_only = subject_chunks - shared_chunks
    answer_only = answer_chunks - shared_chunks

    f: dict[str, float] = {}
    # Raw separated counts (the paper only fixes subject+answer combined).
    f["subject_num_chunks"] = len(subject_chunks)
    f["answer_num_chunks"] = len(answer_chunks)
    f["num_shared_chunks"] = len(shared_chunks)
    f["subject_only_chunks"] = len(subject_only)
    f["answer_only_chunks"] = len(answer_only)
    f["s_pop"] = row.get("s_pop", np.nan)
    f["o_pop"] = row.get("o_pop", np.nan)

    # Interference realized by this checkpoint (subject-only appearances seen).
    subj_only_steps = schedule.steps_for_chunks(subject_only, chunk_to_steps)
    ans_only_steps = schedule.steps_for_chunks(answer_only, chunk_to_steps)
    f["subject_interference_seen"] = sum(s <= checkpoint_step for s in subj_only_steps)
    f["answer_interference_seen"] = sum(s <= checkpoint_step for s in ans_only_steps)

    # Interference AFTER the fact was learned -- the overwriting pressure.
    if first_learned_step is not None and not np.isnan(first_learned_step):
        f["subject_interference_post_learn"] = sum(
            first_learned_step < s <= checkpoint_step for s in subj_only_steps)
        f["answer_interference_post_learn"] = sum(
            first_learned_step < s <= checkpoint_step for s in ans_only_steps)
    else:
        f["subject_interference_post_learn"] = np.nan
        f["answer_interference_post_learn"] = np.nan
    return f
