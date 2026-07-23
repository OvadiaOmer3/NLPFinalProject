"""Tests for text.mention_features against a fixture in LMEnt's real schema.

The fixture mirrors ``DATASET[chunk_id]["metadata"]["entities"]`` exactly as
produced by ``aggregate_entity_annotations`` and read by
``retrieval-index/create_es_index.py`` (char offsets, Wikidata-QID candidates,
per-source scores).  We can't download the 212 GB corpus here, so we drive the
feature logic through DictMentionProvider with hand-built chunks.

Run:  python NLPFinalProject/tests/test_mention_features.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import text as T


def _cand(qid, name="", agg=1.0, hyper=0.0, el=0.0, coref=0.0, coref_cluster=0.0):
    return {"qid": qid, "name": name, "aggregated_score": agg,
            "scores_by_source": {"hyperlinks": hyper, "entity_linking": el,
                                 "coref": coref, "coref_cluster": coref_cluster}}


def _mention(char_start, char_end, text_mention, candidates):
    return {"char_start": char_start, "char_end": char_end,
            "text_mention": text_mention, "candidates": candidates}


SUBJ = "Q42"     # subject entity (s_uri)
OBJ = "Q99"      # answer entity (o_uri)


def approx(a, b):
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, rel_tol=1e-9))


def test_explicit_vs_coref_and_distance():
    # chunk 1: subject via hyperlink (explicit) at chars 0-8; answer (EL) at 20-28.
    # chunk 2: subject via a pronoun / coref at 0-4 (indirect); answer at 40-48.
    chunks = {
        1: {"text": "Ada Lovelace was a mathematician in London.",
            "entities": [
                _mention(0, 8, "Ada", [_cand(SUBJ, "Ada Lovelace", agg=1.0, hyper=1.0)]),
                _mention(20, 28, "mathematician", [_cand(OBJ, "mathematician", agg=0.9, el=0.9)]),
            ]},
        2: {"text": "She was born there; the field she worked in was mathematics.",
            "entities": [
                _mention(0, 3, "She", [_cand(SUBJ, "Ada Lovelace", agg=0.75, coref=0.75)]),
                _mention(40, 48, "the field", [_cand(OBJ, "mathematics", agg=0.5, coref_cluster=0.5)]),
            ]},
    }
    row = {"s_uri": SUBJ, "o_uri": OBJ, "shared_chunks": [1, 2]}
    f = T.mention_features(row, T.DictMentionProvider(chunks))

    assert f["n_subject_mentions"] == 2, f
    # one explicit (hyperlink), one coref (pronoun) -> 0.5 / 0.5
    assert approx(f["frac_explicit_subject"], 0.5), f
    assert approx(f["frac_coref_subject"], 0.5), f
    # distances: chunk1 |4-24|=20, chunk2 |1.5-44|=42.5 ; mean = 31.25
    assert approx(f["subj_answer_distance"], (20.0 + 42.5) / 2), f
    # two different chunk texts -> variety 1.0
    assert approx(f["phrasing_variety"], 1.0), f
    print("test_explicit_vs_coref_and_distance OK", f)


def test_pronoun_is_always_coref():
    # subject candidate has strong hyperlink evidence, but the surface form is a
    # pronoun -> must be classified coref, not explicit.
    chunks = {5: {"text": "They led the team.",
                  "entities": [_mention(0, 4, "They",
                                        [_cand(SUBJ, agg=1.0, hyper=1.0)])]}}
    row = {"s_uri": SUBJ, "o_uri": OBJ, "shared_chunks": [5]}
    f = T.mention_features(row, T.DictMentionProvider(chunks))
    assert f["n_subject_mentions"] == 1
    assert approx(f["frac_coref_subject"], 1.0), f
    assert approx(f["frac_explicit_subject"], 0.0), f
    print("test_pronoun_is_always_coref OK", f)


def test_qid_matching_uses_uri_not_id():
    # A wrong-QID candidate and a below-threshold candidate must NOT match.
    chunks = {7: {"text": "Some text.",
                  "entities": [
                      _mention(0, 3, "foo", [_cand("Q000", agg=1.0, hyper=1.0)]),   # wrong entity
                      _mention(5, 8, "bar", [_cand(SUBJ, agg=0.2, hyper=0.2)]),     # below 0.3
                  ]}}
    row = {"s_uri": SUBJ, "o_uri": OBJ, "shared_chunks": [7]}
    f = T.mention_features(row, T.DictMentionProvider(chunks))
    assert f["n_subject_mentions"] == 0, f
    assert math.isnan(f["frac_explicit_subject"]), f
    print("test_qid_matching_uses_uri_not_id OK", f)


def test_phrasing_variety_repeats():
    same = {"text": "X mentions Y.",
            "entities": [_mention(0, 1, "X", [_cand(SUBJ, agg=1.0, hyper=1.0)])]}
    chunks = {10: same, 11: dict(same), 12: dict(same)}   # identical text x3
    row = {"s_uri": SUBJ, "o_uri": OBJ, "shared_chunks": [10, 11, 12]}
    f = T.mention_features(row, T.DictMentionProvider(chunks))
    assert approx(f["phrasing_variety"], 1 / 3), f     # 1 distinct / 3 total
    assert f["n_subject_mentions"] == 3, f
    print("test_phrasing_variety_repeats OK", f)


def test_empty_shared_chunks():
    row = {"s_uri": SUBJ, "o_uri": OBJ, "shared_chunks": []}
    f = T.mention_features(row, T.DictMentionProvider({}))
    assert f["n_subject_mentions"] == 0
    assert math.isnan(f["phrasing_variety"]) and math.isnan(f["subj_answer_distance"])
    # keys match the placeholder set
    assert set(f) == set(T.MENTION_FEATURE_KEYS) == set(T.empty_mention_features())
    print("test_empty_shared_chunks OK", f)


if __name__ == "__main__":
    test_explicit_vs_coref_and_distance()
    test_pronoun_is_always_coref()
    test_qid_matching_uses_uri_not_id()
    test_phrasing_variety_repeats()
    test_empty_shared_chunks()
    print("\nAll mention_features tests passed.")
