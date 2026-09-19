"""Build OLMo-core's dataset caches for a corpus, on CPU, before training.

The first time OLMo-core opens a corpus it must:
  * build a byte-offset index over each part's metadata CSV (reads ~215 GB),
  * scan each token file for document boundaries and compute chunk boundaries
    that never split an entity mention (`bucketed-doc-indices-*.npy`),
  * compute instance lengths and the VSL sequence-length buckets.

That is hours of single-node CPU work, and `NumpyDataLoaderConfig.build()`
triggers it inside the training job -- with the GPUs sitting idle. Running it
here first means training starts immediately.

The caches land in the corpus's own `dataset-cache/`, keyed by a hash of the
dataset settings, so this MUST use the same config the training run will use --
which is why it reuses train_explicit.build_config() on the same config.json.

Usage (CPU job, needs the lment_train env):
    python prebuild_dataset_cache.py data/runs/<name>/config.json
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: config.py, src/

# prepare() consults the distributed rank/barrier helpers; give them a
# single-process world so they no-op instead of failing.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

import train_explicit  # noqa: E402  (sets up the olmo_core import path)


def main(config_path: str):
    with open(config_path) as f:
        config_dict = json.load(f)
    config_dict.pop("load_path", None)
    config = train_explicit.build_config(config_dict)

    ds_cfg = config.dataset
    print(f"[prebuild] paths      : {ds_cfg.paths}")
    print(f"[prebuild] work_dir   : {ds_cfg.work_dir}")
    print(f"[prebuild] max_seq_len: {ds_cfg.sequence_length}")
    print("[prebuild] building dataset object...", flush=True)

    dataset = ds_cfg.build()
    if not dataset.work_dir_set:
        dataset.work_dir = Path(ds_cfg.work_dir)

    print("[prebuild] preparing caches -- this is the slow part "
          "(index + document boundaries + VSL buckets)", flush=True)
    dataset.prepare()

    print(f"[prebuild] DONE. dataset has {len(dataset):,} instances "
          f"(chunks) at max_sequence_length={dataset.max_sequence_length}")
    work = Path(ds_cfg.work_dir)
    for sub in sorted(p for p in work.rglob("*") if p.is_file()):
        try:
            size = sub.stat().st_size
        except OSError:
            continue
        if size > 1_000_000:
            print(f"    {size/1e6:10.1f} MB  {sub.relative_to(work)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
