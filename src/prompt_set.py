"""Pre-rendered few-shot prompts: build once, evaluate many models against them.

Why this exists
---------------
``FewShot`` selects and renders demonstrations *at evaluation time*, so every run
re-derives its prompts from the live dataset and the flags on the command line.
That is fine for a single sweep, but it makes cross-model comparison fragile: two
runs are only comparable if their ``--num-shots`` / ``--fewshot-mode`` /
``--fewshot-source`` / ``--fewshot-per-prop`` flags matched *and* nothing about
the selection rule (thresholds in ``src/fewshot.py``, the dataset itself) changed
in between.  Nothing records which prompts a parquet was actually produced from
beyond the filename tag.

So: ``build_prompts.py`` materialises the prompts ONCE into a parquet, and the
eval scripts take that file with ``--prompts``.  The file fixes, byte for byte:

  * which facts are evaluated (the pool's subjects are already dropped, and any
    ``--min-shared-chunks`` / ``--limit`` filtering already applied),
  * the order they are evaluated in (so hidden-state dumps line up),
  * the exact prompt string each fact is shown, and how many demonstrations that
    turned out to be (``num_shots`` is per fact -- under ``only-same-prop`` a
    query whose answer knocks out part of its relation's bench gets fewer),
  * which demonstrations those were (``demo_ids``), for inspection.

Everything needed to reproduce or audit the file -- the flags, the demonstration
pool, the dataset id, a hash over the prompt strings -- is stored in the parquet's
schema metadata under ``fewshot_prompts`` and mirrored to a human-readable
``*.meta.json`` sidecar.  Compare two runs' ``prompt_hash`` to prove they saw
identical prompts.

Format
------
One row per evaluated fact::

    id         the popqa-kas fact id
    prop_id    its relation (kept for grouping in analysis)
    num_shots  demonstrations this fact actually got
    demo_ids   ids of those demonstrations, in order
    prompt     the full prompt: demonstration block + this fact's cloze

Metadata (schema key ``fewshot_prompts``) carries ``tag``, which is the artifact
filename suffix the eval scripts use, so a prompt file produced with the same
flags lands on exactly the same output filenames as the old inline path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import config
from src import lment_popqa
from src.fewshot import FewShot

SCHEMA_VERSION = 1
META_KEY = b"fewshot_prompts"
COLUMNS = ("id", "prop_id", "num_shots", "demo_ids", "prompt")


def prompt_dir() -> Path:
    """Default output directory: ``config.PROMPTS_DIR`` (``<data>/artifacts/prompts``)."""
    return Path(config.PROMPTS_DIR)


def fewshot_tag(num_shots: int, mode: str, source: str = "ours") -> str:
    """Artifact filename suffix for a few-shot setting (``""`` at 0-shot).

    Single-sourced here because both the prompt file and the recall parquets it
    produces have to agree on it -- the eval scripts' ``exists -> skip`` guard is
    only safe if settings that change the prompt also change the filename.  The
    exact strings are the ones the existing artifacts already carry (notably
    ``_sameprop`` for ``fixed-block``), so renaming nothing reruns.
    """
    if num_shots <= 0:
        return ""
    tag = f"_{num_shots}shot"
    if source != "ours":
        tag += f"_{source}"
    if mode == "fixed-block":
        tag += "_sameprop"
    elif mode == "only-same-prop":
        tag += "_onlysameprop"
    return tag


class PromptSet:
    """The prompts an eval run is to use, keyed by fact id.

    Built either from a ``FewShot`` (``PromptSet.build``, what the eval scripts
    fall back to when no ``--prompts`` file is given) or read back from a parquet
    (``PromptSet.load``).  Both give the eval loop the same tiny interface:
    ``prompt(id)``, ``shots(id)`` and ``select(df)``.
    """

    def __init__(self, table, meta: dict):
        missing = [c for c in COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"prompt table is missing columns {missing}")
        self.table = table.reset_index(drop=True)
        self.meta = dict(meta)
        self._prompt = dict(zip(self.table["id"], self.table["prompt"]))
        self._shots = dict(zip(self.table["id"], self.table["num_shots"]))
        if len(self._prompt) != len(self.table):
            raise ValueError("prompt table has duplicate fact ids")

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, df, fewshot: FewShot, source: str = "ours",
              extra_meta: dict | None = None) -> "PromptSet":
        """Render every fact in ``df`` through ``fewshot``.

        ``df`` must already be the eval set (pool subjects dropped, filters
        applied); the resulting set is exactly its rows, in its order.

        Note this is also cheaper than rendering inside the eval loop: there,
        ``prefix_for`` and ``shots_for`` each call ``demos_for``, so the
        (per-query, under ``only-same-prop``) demo selection ran twice per fact
        *per checkpoint*.  Here it runs once, full stop.
        """
        import pandas as pd

        rows = []
        for _, r in df.iterrows():
            demos = fewshot.demos_for(r)
            prefix = fewshot.prefix_for_demos(demos)   # cached by demo-id tuple
            rows.append({
                "id": r["id"],
                "prop_id": r["prop_id"],
                "num_shots": len(demos),
                "demo_ids": [d["id"] for d in demos],
                "prompt": lment_popqa.make_prompt(r, prefix=prefix),
            })
        table = pd.DataFrame(rows, columns=list(COLUMNS))

        meta = {
            "schema_version": SCHEMA_VERSION,
            "dataset": getattr(config, "POPQA_DATASET_ID", None),
            "tag": fewshot_tag(fewshot.num_shots, fewshot.mode, source),
            "num_shots": fewshot.num_shots,          # requested
            "fewshot_mode": fewshot.mode,
            "fewshot_source": source,
            "n_facts": len(table),
            "n_distinct_demo_sets": int(table["demo_ids"].map(tuple).nunique()),
            "shots_min": int(table["num_shots"].min()) if len(table) else 0,
            "shots_mean": float(table["num_shots"].mean()) if len(table) else 0.0,
            # The held-out pool: its subjects are the facts NOT in this set.
            "pool_size": int(len(fewshot.demos)),
            "pool_ids": [_json_safe(v) for v in fewshot.demos["id"].tolist()],
            "pool_subject_ids": [_json_safe(v) for v in sorted(fewshot.subject_ids)],
        }
        meta.update(extra_meta or {})
        return cls(table, meta)

    @classmethod
    def load(cls, path) -> "PromptSet":
        import pyarrow.parquet as pq

        path = Path(path)
        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(META_KEY)
        meta = json.loads(raw.decode()) if raw else {}
        if not meta:
            # Readable, but nothing downstream can check what it was built from.
            print(f"[prompts] WARNING: {path.name} carries no fewshot_prompts "
                  "metadata; output filenames fall back to the file's stem")
        ps = cls(table.to_pandas(), meta)
        ps.meta.setdefault("source_path", str(path))
        return ps

    # ------------------------------------------------------------------ #
    # Use
    # ------------------------------------------------------------------ #
    @property
    def ids(self) -> list:
        return self.table["id"].tolist()

    @property
    def tag(self) -> str:
        """Artifact filename suffix this prompt set implies."""
        if "tag" in self.meta:
            return self.meta["tag"]
        # No metadata (hand-made file): fall back to the filename so runs off
        # different prompt files still cannot collide.
        stem = Path(self.meta.get("source_path", "prompts")).stem
        return "_" + stem.replace("prompts_", "").replace("prompts", "custom")

    def prompt(self, fact_id) -> str:
        try:
            return self._prompt[fact_id]
        except KeyError:
            raise KeyError(f"fact {fact_id!r} is not in this prompt set") from None

    def shots(self, fact_id) -> int:
        return int(self._shots[fact_id])

    def select(self, df):
        """Restrict ``df`` to this prompt set's facts, in its order.

        The prompt file -- not the eval script's filters -- defines the eval set,
        so anything it names must be present and anything it doesn't is dropped.
        """
        want = self.ids
        have = set(df["id"])
        missing = [i for i in want if i not in have]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(want)} facts in the prompt set are absent "
                f"from the dataset (first few: {missing[:5]}). The prompt file was "
                f"built against {self.meta.get('dataset')!r}; is the dataset the same?"
            )
        order = {fid: k for k, fid in enumerate(want)}
        out = df[df["id"].isin(set(order))].copy()
        return (out.sort_values("id", key=lambda s: s.map(order))
                   .reset_index(drop=True))

    def prompt_hash(self) -> str:
        """Content hash over (id, prompt) pairs -- equal hashes mean two runs saw
        byte-identical prompts on an identical fact set, in the same order."""
        h = hashlib.sha256()
        for fid, prompt in zip(self.table["id"], self.table["prompt"]):
            h.update(f"{fid}\x01{prompt}\x00".encode("utf-8"))
        return h.hexdigest()

    def head(self, n: int) -> "PromptSet":
        """First ``n`` facts -- a strict subset, for smoke tests."""
        return PromptSet(self.table.head(n).copy(),
                         dict(self.meta, n_facts=min(n, len(self.table)),
                              subset_of=self.meta.get("prompt_hash")))

    # ------------------------------------------------------------------ #
    # I/O + reporting
    # ------------------------------------------------------------------ #
    def save(self, path) -> Path:
        """Write the parquet (metadata embedded) plus a ``*.meta.json`` sidecar.

        The embedded copy is the one ``load`` reads; the sidecar is for humans and
        for grepping a directory of prompt files.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta, prompt_hash=self.prompt_hash())
        t = pa.Table.from_pandas(self.table, preserve_index=False)
        md = dict(t.schema.metadata or {})
        md[META_KEY] = json.dumps(meta, default=str).encode()
        pq.write_table(t.replace_schema_metadata(md), path)
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")
        self.meta = meta
        return path

    def describe(self) -> str:
        m = self.meta
        lines = [
            f"[prompts] {m.get('n_facts', len(self.table))} facts | "
            f"num_shots={m.get('num_shots')} mode={m.get('fewshot_mode')} "
            f"source={m.get('fewshot_source')} tag={self.tag!r}",
            f"[prompts] shots per fact: min={int(self.table['num_shots'].min())} "
            f"mean={self.table['num_shots'].mean():.2f} "
            f"max={int(self.table['num_shots'].max())}; "
            f"{int(self.table['demo_ids'].map(tuple).nunique())} distinct demo sets",
        ]
        if m.get("prompt_hash"):
            lines.append(f"[prompts] prompt_hash={m['prompt_hash'][:16]}")
        return "\n".join(lines)


def _json_safe(v):
    """numpy scalars -> Python scalars, so ``json.dumps`` accepts the pool ids."""
    return v.item() if hasattr(v, "item") else v


# --------------------------------------------------------------------------- #
# CLI  (thin wrappers in feature_extraction/ and explicit_mentions/ call this)
# --------------------------------------------------------------------------- #
def build_argparser(prog: str | None = None) -> argparse.ArgumentParser:
    from src import fewshot as fewshot_mod

    ap = argparse.ArgumentParser(
        prog=prog, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-shots", type=int, default=config.NUM_SHOTS,
                    help=f"demonstrations per fact (default {config.NUM_SHOTS}); "
                         "0 = bare zero-shot cloze")
    ap.add_argument("--fewshot-mode", choices=fewshot_mod.MODES,
                    default="drop-same-prop",
                    help="'drop-same-prop' (default): the pool minus the query's "
                         "own relation. 'fixed-block': one constant block, "
                         "upstream's behaviour. 'only-same-prop': demos of the "
                         "query's own relation only, none that would hand over its "
                         "answer. See src/fewshot.py.")
    ap.add_argument("--fewshot-source", choices=("ours", "upstream"), default="ours",
                    help="'ours' selects by the rule in src/fewshot.py; 'upstream' "
                         "uses LMEnt's verbatim demos (leaky; parity runs only)")
    ap.add_argument("--fewshot-per-prop", type=int, default=None,
                    help="demonstrations selected per prop_id (mode-dependent "
                         "default; see src/fewshot.py)")
    ap.add_argument("--min-shared-chunks", type=int, default=0,
                    help="drop facts whose subject and answer never co-occur in a "
                         "shared_chunk. Baked into the prompt set, so every model "
                         "evaluated off this file scores the same facts.")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only the first N facts (smoke test)")
    ap.add_argument("--out", type=str, default=None,
                    help="output parquet path (default: <data>/artifacts/prompts/"
                         "prompts{tag}.parquet)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="directory for the default filename (default: "
                         f"{prompt_dir()})")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing prompt file instead of refusing")
    ap.add_argument("--show", type=int, default=0,
                    help="print the first N prompts after building")
    ap.add_argument("--inspect", type=str, default=None,
                    help="print an existing prompt file's metadata and exit "
                         "(builds nothing)")
    return ap


def _inspect(path: str) -> None:
    ps = PromptSet.load(path)
    print(json.dumps(ps.meta, indent=2, default=str))
    print(ps.describe())
    print(f"[prompts] recomputed prompt_hash={ps.prompt_hash()[:16]} "
          f"(stored {str(ps.meta.get('prompt_hash'))[:16]})")


def main(argv=None) -> None:
    from src import data as D

    ap = build_argparser()
    args = ap.parse_args(argv)

    if args.inspect:
        _inspect(args.inspect)
        return

    tag = fewshot_tag(args.num_shots, args.fewshot_mode, args.fewshot_source)
    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path(args.out_dir) if args.out_dir else prompt_dir()
        out_path = out_dir / f"prompts{tag or '_0shot'}.parquet"
    if out_path.exists() and not args.force:
        ap.error(f"{out_path} exists; pass --force to overwrite (prompt files are "
                 "meant to be stable -- overwriting one silently changes what every "
                 "run that cites it was scored on)")

    df = D.load_popqa_df()
    # Demonstrations are chosen from the FULL dataset, before the filters below, so
    # the pool doesn't shift when --limit / --min-shared-chunks change.
    fewshot = FewShot.build(df, num_shots=args.num_shots,
                            source=args.fewshot_source,
                            per_prop=args.fewshot_per_prop,
                            mode=args.fewshot_mode)
    df = fewshot.drop_subjects(df)
    if args.min_shared_chunks > 0:
        n_before = len(df)
        n_shared = df["shared_chunks"].apply(lambda c: len(c) if c is not None else 0)
        df = df[n_shared >= args.min_shared_chunks].reset_index(drop=True)
        print(f"[filter] dropped {n_before - len(df)} facts with "
              f"< {args.min_shared_chunks} shared_chunks ({len(df)} remain)")
    if args.limit:
        df = df.head(args.limit)

    ps = PromptSet.build(df, fewshot, source=args.fewshot_source, extra_meta={
        "fewshot_per_prop": args.fewshot_per_prop,
        "min_shared_chunks": args.min_shared_chunks,
        "limit": args.limit,
    })
    ps.save(out_path)
    print(ps.describe())
    for prompt in ps.table["prompt"].head(max(args.show, 0)):
        print("-" * 72)
        print(prompt)
    print("-" * 72 if args.show else "")
    print(f"[write] {out_path}")
    print(f"[write] {out_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
