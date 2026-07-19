"""Group 3 - "What the text itself looks like".

Some of these are cheap (answer token length, relation type).  The mention-
based ones (explicit vs. coref, subject-answer distance, phrasing variety) need
LMEnt's per-chunk annotations, which live in the ``dataset-metadata`` layer
(ReFinED entity links + Maverick coreference).  We express those through a thin
``MentionProvider`` so the exact file format can be wired up in one place.

A ``Mention`` is: (chunk_id, entity_id, char_start, char_end, source)
where source in {"refined", "maverick"} -> {explicit, coreference}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mention:
    chunk_id: int
    entity_id: int
    start: int          # token index within the chunk
    end: int
    source: str         # "refined" (hyperlink/EL, explicit) | "maverick" (coref)


class MentionProvider:
    """Abstract access to per-chunk mentions and chunk text.

    Implement against LMEnt's dataset-metadata / dataset-tokenized files.
    VERIFY the concrete schema (column names, whether offsets are token or char).
    """

    def mentions_in_chunk(self, chunk_id: int) -> list[Mention]:
        raise NotImplementedError

    def chunk_text(self, chunk_id: int) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Cheap features (no external data needed)
# --------------------------------------------------------------------------- #
def cheap_text_features(row, tokenizer) -> dict:
    f = {}
    f["relation"] = row["prop"]                       # relation type (categorical)
    f["answer_token_len"] = len(tokenizer(" " + row["obj"],
                                          add_special_tokens=False)["input_ids"])
    f["answer_char_len"] = len(row["obj"])
    f["answer_word_len"] = len(row["obj"].split())
    return f


# --------------------------------------------------------------------------- #
# Mention-based features (need MentionProvider wired to LMEnt metadata)
# --------------------------------------------------------------------------- #
def mention_features(row, provider: MentionProvider) -> dict:
    """Explicitness, subject-answer proximity, and phrasing variety, aggregated
    over the fact's shared_chunks."""
    subj_id = int(row["subj_id"])
    obj_id = int(row["obj_id"])
    shared = [int(c) for c in row["shared_chunks"]]

    n_explicit = n_coref = n_subj_mentions = 0
    distances: list[int] = []
    texts: list[str] = []

    for cid in shared:
        ms = provider.mentions_in_chunk(cid)
        subj_ms = [m for m in ms if m.entity_id == subj_id]
        ans_ms = [m for m in ms if m.entity_id == obj_id]
        for m in subj_ms:
            n_subj_mentions += 1
            if m.source == "refined":
                n_explicit += 1
            else:
                n_coref += 1
        # closest subject-answer token distance within the chunk
        if subj_ms and ans_ms:
            distances.append(min(abs(s.start - a.start)
                                 for s in subj_ms for a in ans_ms))
        texts.append(provider.chunk_text(cid))

    f: dict[str, float] = {}
    # How the fact is written: fraction of explicit (EL) vs. coref mentions.
    if n_subj_mentions:
        f["frac_explicit_subject"] = n_explicit / n_subj_mentions
        f["frac_coref_subject"] = n_coref / n_subj_mentions
    else:
        f["frac_explicit_subject"] = np.nan
        f["frac_coref_subject"] = np.nan
    f["n_subject_mentions"] = n_subj_mentions

    # Subject-answer proximity inside the chunk (mean of per-chunk minima).
    f["subj_answer_distance"] = float(np.mean(distances)) if distances else np.nan

    # Phrasing variety: distinct chunk texts / total (1.0 = all different).
    if texts:
        f["phrasing_variety"] = len(set(texts)) / len(texts)
    else:
        f["phrasing_variety"] = np.nan
    return f
