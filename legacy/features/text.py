"""Group 3 - "What the text itself looks like".

Some of these are cheap (answer token length, relation type).  The mention-
based ones (explicit vs. coref, subject-answer distance, phrasing variety) need
LMEnt's per-chunk entity annotations (ReFinED entity links + Maverick
coreference, aggregated by ``aggregate_entity_annotations``).

Schema is taken from LMEnt's own reader, ``retrieval-index/create_es_index.py``
(``DATASET[chunk_id]["metadata"]["entities"]``).  Each chunk's ``entities`` is a
list of mentions:

    {
      "char_start": int, "char_end": int,      # CHAR offsets into the chunk text
      "text_mention": str,
      "candidates": [
        {"qid": "Q42", "name": "...", "aggregated_score": float,
         "scores_by_source": {"hyperlinks": f, "entity_linking": f,
                              "coref": f, "coref_cluster": f}},
        ...
      ]
    }

Entities are Wikidata **QIDs**.  In popqa-kas those live in ``s_uri`` / ``o_uri``
(e.g. ``"Q5543720"``) -- NOT ``subj_id`` / ``obj_id`` (internal PopQA ints).
"Explicit" evidence = ``hyperlinks`` / ``entity_linking`` (the entity is directly
linked); "coreference" evidence = ``coref`` / ``coref_cluster`` (a pronoun or
description resolved to the entity), which is the weaker training signal.
"""
from __future__ import annotations

import re

import numpy as np

# Minimum aggregated_score for a candidate to count as "this mention refers to
# that entity" (matches the 0.3 threshold LMEnt uses in aggregate_entity_annotations).
MIN_CANDIDATE_SCORE = 0.3

_PRONOUN_RE = re.compile(
    r"\b(?:he|she|it|they|we|i|you|him|her|us|them|me|my|your|his|their|our|its|"
    r"mine|yours|hers|ours|theirs)\b", re.I)


def _is_pronoun(text: str) -> bool:
    return bool(text) and _PRONOUN_RE.fullmatch(text.strip()) is not None


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
# Chunk-metadata providers (access to LMEnt's per-chunk entity annotations)
# --------------------------------------------------------------------------- #
class MentionProvider:
    """Access to a chunk's aggregated entity mentions and its text.

    ``entities_for_chunk`` returns the list described in the module docstring;
    ``chunk_text`` returns the decoded chunk text (or ``None`` if unavailable).
    """

    def entities_for_chunk(self, chunk_id: int) -> list[dict]:
        raise NotImplementedError

    def chunk_text(self, chunk_id: int) -> str | None:
        raise NotImplementedError


class DictMentionProvider(MentionProvider):
    """In-memory provider for tests / small exports.

    ``chunks`` maps ``chunk_id -> {"text": str, "entities": [...]}`` using LMEnt's
    real schema.
    """

    def __init__(self, chunks: dict):
        self._c = {int(k): v for k, v in chunks.items()}

    def entities_for_chunk(self, chunk_id: int) -> list[dict]:
        return self._c.get(int(chunk_id), {}).get("entities", []) or []

    def chunk_text(self, chunk_id: int) -> str | None:
        c = self._c.get(int(chunk_id))
        return c.get("text") if c else None


class OlmoCoreMentionProvider(MentionProvider):
    """Backed by LMEnt's OLMo-core ``NumpyDataset`` -- the documented access path
    in ``retrieval-index/create_es_index.py``.

    ``dataset[chunk_id]`` yields ``{"input_ids", "metadata"}``;
    ``metadata["entities"]`` holds the aggregated ReFinED + Maverick mentions and
    the chunk text is ``tokenizer.decode(input_ids)``.  Requires ``olmo_core`` +
    the tokenized dataset, so this is only for a machine that downloaded the full
    corpus (the PopQA-only analysis does not need it).  Build it with
    :func:`build_olmo_core_provider`.
    """

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self._cache: dict[int, dict] = {}

    def _get(self, chunk_id: int) -> dict:
        cid = int(chunk_id)
        if cid not in self._cache:
            self._cache[cid] = self.dataset[cid]
        return self._cache[cid]

    def entities_for_chunk(self, chunk_id: int) -> list[dict]:
        return self._get(chunk_id)["metadata"].get("entities", []) or []

    def chunk_text(self, chunk_id: int) -> str | None:
        ids = self._get(chunk_id)["input_ids"]
        return self.tokenizer.decode(ids.tolist() if hasattr(ids, "tolist") else ids)


def build_olmo_core_provider(glob_pattern: str, work_dir: str,
                             max_sequence_length: int = 2048,
                             min_sequence_length: int = 64):
    """Construct an :class:`OlmoCoreMentionProvider` following LMEnt's
    ``create_es_index.py`` (VSL curriculum, dolma2 tokenizer).  Returns ``None``
    if ``olmo_core`` / ``olmo_eval`` aren't installed so callers can degrade
    gracefully.  ``glob_pattern`` points at the tokenized ``*.npy`` files.
    """
    try:
        from olmo_core.data import (NumpyDatasetConfig, NumpyDatasetType,
                                    TokenizerConfig)
        from olmo_core.data.numpy_dataset import (VSLCurriculumConfig,
                                                  VSLCurriculumType)
        from olmo_eval import HFTokenizer
    except Exception:
        return None

    tok_cfg = TokenizerConfig.dolma2()
    ds = NumpyDatasetConfig.glob(
        glob_pattern,
        name=NumpyDatasetType.kas_vsl,
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2,
                                           num_cycles=8, balanced=False),
        tokenizer=tok_cfg,
        work_dir=work_dir,
        include_instance_metadata=True,
    ).build()
    tokenizer = HFTokenizer(tok_cfg.identifier, pad_token_id=tok_cfg.pad_token_id,
                            eos_token_id=tok_cfg.eos_token_id,
                            bos_token_id=tok_cfg.bos_token_id)
    return OlmoCoreMentionProvider(ds, tokenizer)


# --------------------------------------------------------------------------- #
# Mention-based features
# --------------------------------------------------------------------------- #
MENTION_FEATURE_KEYS = ("frac_explicit_subject", "frac_coref_subject",
                        "n_subject_mentions", "subj_answer_distance",
                        "phrasing_variety")


def empty_mention_features() -> dict:
    """NaN/0 placeholder so the columns exist even without a provider."""
    return {"frac_explicit_subject": np.nan, "frac_coref_subject": np.nan,
            "n_subject_mentions": 0, "subj_answer_distance": np.nan,
            "phrasing_variety": np.nan}


def _match_candidate(entity: dict, qid: str, min_score: float):
    """Highest-scoring candidate of ``entity`` that refers to ``qid`` above
    ``min_score``, or ``None``."""
    if not qid:
        return None
    best = None
    for c in entity.get("candidates", []):
        if c.get("qid") == qid and float(c.get("aggregated_score", 0.0)) >= min_score:
            if best is None or c["aggregated_score"] > best["aggregated_score"]:
                best = c
    return best


def _is_explicit(candidate: dict, text_mention: str) -> bool:
    """Explicit (entity-link / hyperlink) vs. coreference, from the matched
    candidate's per-source evidence.  Pronoun surface forms are always coref."""
    if _is_pronoun(text_mention):
        return False
    s = candidate.get("scores_by_source", {}) or {}
    explicit = float(s.get("hyperlinks", 0.0)) + float(s.get("entity_linking", 0.0))
    coref = float(s.get("coref", 0.0)) + float(s.get("coref_cluster", 0.0))
    return explicit > 0.0 and explicit >= coref


def _span_mid(entity: dict) -> float:
    return (float(entity.get("char_start", 0)) + float(entity.get("char_end", 0))) / 2.0


def mention_features(row, provider: MentionProvider,
                     min_score: float = MIN_CANDIDATE_SCORE) -> dict:
    """Explicitness, subject-answer proximity, and phrasing variety over the
    fact's ``shared_chunks``.

    Entities are matched by Wikidata QID: subject = ``row["s_uri"]``,
    answer = ``row["o_uri"]``.  ``subj_answer_distance`` is a **character**
    distance (mention midpoints) inside the chunk -- the aggregated metadata
    stores char offsets, not token indices.
    """
    subj_qid = str(row.get("s_uri") or "")
    obj_qid = str(row.get("o_uri") or "")
    shared = [int(c) for c in (row.get("shared_chunks") or [])]

    n_explicit = n_coref = n_subj_mentions = 0
    distances: list[float] = []
    texts: list[str] = []

    for cid in shared:
        entities = provider.entities_for_chunk(cid)
        text = provider.chunk_text(cid)
        if text is not None:
            texts.append(text)

        subj_mids, obj_mids = [], []
        for e in entities:
            sc = _match_candidate(e, subj_qid, min_score)
            if sc is not None:
                n_subj_mentions += 1
                if _is_explicit(sc, e.get("text_mention", "")):
                    n_explicit += 1
                else:
                    n_coref += 1
                subj_mids.append(_span_mid(e))
            if _match_candidate(e, obj_qid, min_score) is not None:
                obj_mids.append(_span_mid(e))

        # closest subject-answer mention distance within the chunk (chars)
        if subj_mids and obj_mids:
            distances.append(min(abs(s - o) for s in subj_mids for o in obj_mids))

    f: dict[str, float] = {}
    if n_subj_mentions:
        f["frac_explicit_subject"] = n_explicit / n_subj_mentions
        f["frac_coref_subject"] = n_coref / n_subj_mentions
    else:
        f["frac_explicit_subject"] = np.nan
        f["frac_coref_subject"] = np.nan
    f["n_subject_mentions"] = n_subj_mentions
    f["subj_answer_distance"] = float(np.mean(distances)) if distances else np.nan
    # Phrasing variety: distinct chunk texts / total (1.0 = all different).
    f["phrasing_variety"] = len(set(texts)) / len(texts) if texts else np.nan
    return f
