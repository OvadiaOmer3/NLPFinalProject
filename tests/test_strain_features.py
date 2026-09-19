"""Tests for features.strain against hand-built fixtures.

The fixtures encode the three mistakes that actually happened while developing
this metric, so a regression re-introduces a failing test rather than a
plausible-looking number:

  * two distinct entities sharing a surface form must count as TWO competitors
    (the "Buffalo city / Buffalo team" case);
  * an entity must never compete with itself;
  * a fact's own object is not a distractor for its own subject.

Run:  python NLPFinalProject/tests/test_strain_features.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import strain as S


# --------------------------------------------------------------------------- #
# Fixture: 4 facts, deliberately containing a same-string/different-QID pair
# --------------------------------------------------------------------------- #
#   Q1 "Buffalo Bills"   (team)    chunks 1,2,3
#   Q2 "Buffalo"         (city)    chunks 2,3,4      <- shares token "buffalo"
#   Q3 "Josh Allen"                chunks 1,5
#   Q4 "New York"                  chunks 3,4,6
#   Q5 "Chicago"                   chunks 7
FACTS = pd.DataFrame([
    # subject Q3 -> object Q1 ("plays for"), shared chunk = {1}
    dict(id=1, prop="team", subj="Josh Allen", obj="Buffalo Bills",
         s_uri="Q3", o_uri="Q1",
         subject_chunks=[1, 5], answer_chunks=[1, 2, 3], shared_chunks=[1]),
    # subject Q1 -> object Q2 ("based in"), shared chunks = {2,3}.
    # Chunk 3 also holds Q4, a different valid "city" answer -> 1 distractor,
    # and that same chunk is a shared chunk: the two are not exclusive.
    dict(id=2, prop="city", subj="Buffalo Bills", obj="Buffalo",
         s_uri="Q1", o_uri="Q2",
         subject_chunks=[1, 2, 3], answer_chunks=[2, 3, 4], shared_chunks=[2, 3]),
    # subject Q2 -> object Q4.  Q2 is ITSELF a "city" answer (fact 2), so this
    # row is the self-distractor regression case.
    dict(id=3, prop="city", subj="Buffalo", obj="New York",
         s_uri="Q2", o_uri="Q4",
         subject_chunks=[2, 3, 4], answer_chunks=[3, 4, 6], shared_chunks=[3, 4]),
    # subject Q3 -> object Q5, no shared chunks at all
    dict(id=4, prop="city", subj="Josh Allen", obj="Chicago",
         s_uri="Q3", o_uri="Q5",
         subject_chunks=[1, 5], answer_chunks=[7], shared_chunks=[]),
])


def universe():
    return S.EntityUniverse.from_popqa(FACTS)


def index(df_cutoff=None, idf=False):
    u = universe()
    return u, S.TokenIndex(u, u.tokens(), df_cutoff=df_cutoff, idf_weighted=idf)


def test_universe_dedupes_entities_and_chunks():
    u = universe()
    assert len(u) == 5, len(u)
    assert u.qid_to_form["Q1"] == "Buffalo Bills"
    assert list(u.chunks["Q2"]) == [2, 3, 4]
    # Q1 appears as an object (fact 1) and a subject (fact 2) with the same chunks
    assert list(u.chunks["Q1"]) == [1, 2, 3]


def test_inconsistent_chunk_lists_are_rejected():
    recs = FACTS.to_dict("records")
    recs[1]["subject_chunks"] = [1, 2, 3, 99]     # Q1 disagrees with fact 1's answer_chunks
    bad = pd.DataFrame(recs)
    try:
        S.EntityUniverse.from_popqa(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError on inconsistent chunk lists")


def test_same_string_different_qid_are_two_competitors():
    """The whole point of counting QIDs. 'Buffalo' the city and 'Buffalo Bills'
    the team share the token 'buffalo' and must both be reachable."""
    u, idx = index()
    assert idx.token_to_qids["buffalo"] == {"Q1", "Q2"}
    # Q1's rivals include Q2 via "buffalo" -- and NOT Q1 itself
    assert idx.rivals("Q1") == {"Q2"}
    assert "Q1" not in idx.rivals("Q1")


def test_strain_excludes_the_facts_own_other_endpoint():
    u, idx = index()
    # fact 2 is Q1 -> Q2.  Q1's only rival is Q2, which is its own answer,
    # so strain on the subject side is 0, not 1.
    assert idx.strain("Q1") == 1.0
    assert idx.strain("Q1", exclude={"Q2"}) == 0.0


def test_df_cutoff_drops_shared_tokens():
    u, idx_all = index(df_cutoff=None)
    assert idx_all.rivals("Q1") == {"Q2"}
    # "buffalo" is carried by 2 of 5 entities; a cutoff of 1 removes it
    _, idx_cut = index(df_cutoff=1)
    assert idx_cut.rivals("Q1") == set()


def test_idf_weighting_downweights_common_tokens():
    _, plain = index(idf=False)
    _, weighted = index(idf=True)
    # one rival either way, but the weighted version scores it below 1
    assert plain.strain("Q1") == 1.0
    assert 0.0 < weighted.strain("Q1") < 1.0


def test_rarest_token_is_the_discriminative_one():
    u, idx = index()
    # Q1 = "Buffalo Bills": "buffalo" is in 2 entities, "bills" in 1
    assert idx.rarest_token("Q1") == "bills"
    assert idx.n_rivals_on("bills") == 0
    assert idx.n_rivals_on("buffalo") == 1


def test_ownership_share():
    u, idx = index()
    # token "buffalo" is carried by Q1 (chunks 1,2,3) and Q2 (chunks 2,3,4)
    # -> pool = {1,2,3,4} = 4 chunks
    assert idx.token_chunk_pool("buffalo") == 4
    # fact 2 (Q1->Q2) has 2 shared chunks, so it owns 2/4 of the "buffalo" evidence
    own = idx.ownership("Q1", n_shared=2)
    assert own["rivals_rare"] == 0            # rarest token "bills" has no rivals
    assert np.isclose(idx.token_chunk_pool("bills"), 3)   # only Q1's chunks
    assert np.isclose(own["own_rare"], 2 / 3)


def test_alt_answer_counts_only_other_valid_answers():
    u = universe()
    out = S.alt_answer_features(FACTS, u, alpha=0.0)
    # fact 2: subject Q1 (chunks 1,2,3), true answer Q2 (chunks 2,3,4).
    # Other "city" answers are Q4 (chunks 3,4,6) and Q5 (chunk 7).
    #   chunk 1 -> nothing
    #   chunk 2 -> holds Q2, the true answer                     => shared
    #   chunk 3 -> holds Q2 (shared) AND Q4 (a rival answer)     => both
    assert out[2]["shared"] == 2, out[2]
    assert out[2]["distract"] == 1, out[2]
    assert out[2]["evidence"] == 3
    assert np.isclose(out[2]["ratio"], 2 / 3)


def test_subject_is_not_its_own_distractor():
    """Q2 ("Buffalo") is the answer to fact 2 and the subject of fact 3.  Every
    chunk of a subject trivially contains that subject, so without excluding it
    `distract` degenerates into "number of subject chunks"."""
    u = universe()
    out = S.alt_answer_features(FACTS, u, alpha=0.0)
    # fact 3: subject Q2 (chunks 2,3,4), true answer Q4 (chunks 3,4,6).
    # The only other "city" answer is Q5 (chunk 7), which never co-occurs.
    assert out[3]["shared"] == 2, out[3]
    assert out[3]["distract"] == 0, out[3]


def test_ratio_is_undefined_without_evidence():
    u = universe()
    out = S.alt_answer_features(FACTS, u, alpha=0.0)
    # fact 4's subject never appears with any candidate answer: there is no
    # signal to be clean or dirty, so the ratio must not default to 1.0.
    assert out[4]["evidence"] == 0
    assert np.isnan(out[4]["ratio"])


def test_ratio_smoothing_protects_thin_evidence():
    u = universe()
    raw = S.alt_answer_features(FACTS, u, alpha=0.0)
    sm = S.alt_answer_features(FACTS, u, alpha=1.0)
    # fact 1: shared=1, distract=0 -> a perfect 1.0 raw, which is the artifact
    # that made "cleanest ratio" fill up with no-evidence facts.
    assert np.isclose(raw[1]["ratio"], 1.0)
    assert sm[1]["ratio"] < 0.75


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def _fake_batch_indices(tmp, per_epoch_orders):
    """Write one batch_indices file per epoch; each is a list of 1-chunk steps."""
    paths = []
    for e, order in enumerate(per_epoch_orders, start=1):
        p = tmp / f"bi_epoch_{e}.npy"
        np.save(p, np.array([np.asarray([c]) for c in order], dtype=object),
                allow_pickle=True)
        paths.append(p)
    return paths


def test_window_map_across_epochs():
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    # 2 epochs x 4 steps.  Global steps 0..7.
    #   epoch 1 order: chunks 1,2,3,4   -> global steps 0,1,2,3
    #   epoch 2 order: chunks 4,3,2,1   -> global steps 4,5,6,7
    paths = _fake_batch_indices(tmp, [[1, 2, 3, 4], [4, 3, 2, 1]])
    # boundaries 0,2,4,6,8 -> windows (0,2],(2,4],(4,6],(6,8]
    wm = S.WindowMap.from_batch_indices(paths, [0, 2, 4, 6, 8], steps_per_epoch=4)
    assert wm.n_windows == 4

    # chunk 1: epoch1 step 0 (excluded, == boundary[0]), epoch2 step 7 -> window 3
    assert list(wm.in_window([1], 0)) == [False]
    assert list(wm.in_window([1], 3)) == [True]
    # chunk 3: epoch1 step 2 -> window 0 ; epoch2 step 5 -> window 2
    assert list(wm.in_window([3], 0)) == [True]
    assert list(wm.in_window([3], 2)) == [True]
    assert list(wm.in_window([3], 1)) == [False]
    # a chunk id beyond the corpus never matches
    assert list(wm.in_window([10 ** 6], 0)) == [False]


def test_windowed_strain_only_counts_live_entities():
    u, idx = index()
    # Q1's only rival is Q2.  If Q2 was not seen this window, strain is 0.
    assert S.windowed_strain(idx, "Q1", active={"Q3", "Q4"}) == 0.0
    assert S.windowed_strain(idx, "Q1", active={"Q2", "Q3"}) == 1.0
    # ...and the fact's own answer still doesn't count
    assert S.windowed_strain(idx, "Q1", active={"Q2"}, exclude={"Q2"}) == 0.0


def test_active_entities_uses_chunk_membership():
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    paths = _fake_batch_indices(tmp, [[1, 2, 3, 4, 5, 6, 7]])
    wm = S.WindowMap.from_batch_indices(paths, [0, 3, 7], steps_per_epoch=7)
    u = universe()
    # window 0 = steps (0,3] -> chunks 2,3,4 (step0 chunk1 is excluded)
    act = wm.active_entities(u, 0)
    assert "Q1" in act          # Q1 has chunks 1,2,3 -> 2 and 3 are in window 0
    assert "Q5" not in act      # Q5 has chunk 7 -> step 6, window 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")
