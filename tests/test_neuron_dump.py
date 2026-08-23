"""Tests for internal.NeuronDump -- the top-k active-MLP-neuron store.

This path had never been exercised (``--top-k`` defaults to None and no run has
used it), so these cover the contract the offline overlap analyses rely on:

  * idx/val stay aligned -- ``val[f, l, j]`` is the activation of ``idx[f, l, j]``
  * unwritten rows keep their sentinels (MISSING for indices, NaN for values)
  * ``resume=True`` reopens in place instead of truncating earlier rows back
  * values are signed and survive the float16 round-trip
  * ``store_values=False`` writes no value array and still works

Run:  python NLPFinalProject/tests/test_neuron_dump.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.internal import NeuronDump

STEP, IDS, NL, K = 4242, [11, 22, 33], 4, 8
POS = ("answer", "subject_last")


def _payload(seed, n_layers=NL, k=K):
    """One fact's neuron record, in extract_fact's return shape."""
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.choice(5632, k, replace=False).astype(np.uint16)
                    for _ in range(n_layers)])
    val = (rng.standard_normal((n_layers, k)) * 3).astype(np.float16)
    return {"idx": idx, "val": val}


def _paths(d, name):
    return (Path(d) / f"neurons_step{STEP}_{name}_idx.npy",
            Path(d) / f"neurons_step{STEP}_{name}_val.npy")


def test_roundtrip_and_alignment():
    d = tempfile.mkdtemp()
    try:
        dump = NeuronDump(STEP, IDS, NL, K, POS, d)
        recs = {0: {n: _payload(hash((n, 0)) % 999) for n in POS},
                2: {n: _payload(hash((n, 2)) % 999) for n in POS}}
        for i, rec in recs.items():
            dump.write(i, rec)
        dump.close()

        for name in POS:
            ip, vp = _paths(d, name)
            assert ip.exists() and vp.exists(), f"missing dump for {name}"
            idx, val = np.load(ip), np.load(vp)
            assert idx.shape == (len(IDS), NL, K) == val.shape
            assert idx.dtype == np.uint16 and val.dtype == np.float16
            for i, rec in recs.items():
                assert np.array_equal(idx[i], rec[name]["idx"]), "index mismatch"
                assert np.array_equal(val[i], rec[name]["val"]), "value mismatch"
            # row 1 was never written -> sentinels intact
            assert (idx[1] == NeuronDump.MISSING).all(), "unwritten idx not sentinel"
            assert np.isnan(val[1]).all(), "unwritten val not NaN"
            # signedness actually survives (the whole point of storing values)
            assert (val[0] < 0).any(), "no negative activations survived"

        ids_saved = np.load(Path(d) / f"neurons_step{STEP}_ids.npy")
        assert ids_saved.tolist() == IDS
        meta = json.loads((Path(d) / f"neurons_step{STEP}_meta.json").read_text())
        assert meta["k"] == K and meta["n_layers"] == NL
        assert meta["stores_values"] is True and meta["values_are_signed"] is True
        assert meta["missing_sentinel"] == NeuronDump.MISSING
        print("ok  roundtrip + idx/val alignment + sentinels + meta")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_resume_preserves_written_rows():
    d = tempfile.mkdtemp()
    try:
        first = {n: _payload(7) for n in POS}
        dump = NeuronDump(STEP, IDS, NL, K, POS, d)
        dump.write(0, first)
        dump.close()

        # a crash here; the rerun reopens with resume=True
        dump2 = NeuronDump(STEP, IDS, NL, K, POS, d, resume=True)
        dump2.write(1, {n: _payload(8) for n in POS})
        dump2.close()

        for name in POS:
            ip, vp = _paths(d, name)
            idx, val = np.load(ip), np.load(vp)
            assert np.array_equal(idx[0], first[name]["idx"]), "resume truncated row 0"
            assert np.array_equal(val[0], first[name]["val"]), "resume truncated values"
            assert (idx[1] != NeuronDump.MISSING).any(), "row 1 not written"
        print("ok  resume reopens in place")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_resume_shape_mismatch_raises():
    d = tempfile.mkdtemp()
    try:
        NeuronDump(STEP, IDS, NL, K, POS, d).close()
        try:
            NeuronDump(STEP, IDS, NL, K * 2, POS, d, resume=True)
        except ValueError as e:
            assert "expected" in str(e)
            print("ok  resume with a different k raises instead of corrupting")
        else:
            raise AssertionError("expected ValueError on k change")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_store_values_false():
    d = tempfile.mkdtemp()
    try:
        dump = NeuronDump(STEP, IDS, NL, K, POS, d, store_values=False)
        dump.write(0, {n: _payload(3) for n in POS})
        dump.close()
        for name in POS:
            ip, vp = _paths(d, name)
            assert ip.exists(), "index array should still be written"
            assert not vp.exists(), "value array should be skipped"
        meta = json.loads((Path(d) / f"neurons_step{STEP}_meta.json").read_text())
        assert meta["stores_values"] is False and meta["val_dtype"] is None
        print("ok  --no-neuron-values writes indices only")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_missing_position_leaves_sentinels():
    d = tempfile.mkdtemp()
    try:
        dump = NeuronDump(STEP, IDS, NL, K, POS, d)
        # subject_last couldn't be located for this fact -> None, per extract_fact
        dump.write(0, {"answer": _payload(5), "subject_last": None})
        dump.close()
        idx_a = np.load(_paths(d, "answer")[0])
        idx_s, val_s = (np.load(p) for p in _paths(d, "subject_last"))
        assert (idx_a[0] != NeuronDump.MISSING).any(), "answer row not written"
        assert (idx_s[0] == NeuronDump.MISSING).all(), "missing position not sentinel"
        assert np.isnan(val_s[0]).all()
        print("ok  a position that couldn't be located stays sentinel")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_roundtrip_and_alignment()
    test_resume_preserves_written_rows()
    test_resume_shape_mismatch_raises()
    test_store_values_false()
    test_missing_position_leaves_sentinels()
    print("\nall NeuronDump tests passed")
