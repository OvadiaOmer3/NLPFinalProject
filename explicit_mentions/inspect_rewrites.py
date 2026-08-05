"""Inspect and verify a rewritten corpus part.

Two subcommands:

  samples  Pretty-print the before/after snippets that build_explicit_corpus.py
           saved (quick eyeball check of replacement quality):
               python inspect_rewrites.py samples --corpus-tag smoke --n 20

  verify   Structural verification of an output part against the source
           (run after the smoke test, before committing to the full rewrite):
             * CSV start/end are contiguous and match the npy token count
             * every doc ends with EOS
             * decoded text reproduces every mention's text_mention at
               [char_start:char_end], and tok_start/tok_end map back to a
               token span covering it
             * rewritten docs actually differ, passthrough docs are verbatim
               python inspect_rewrites.py verify --corpus-tag smoke --part 0 --n-docs 500
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from build_explicit_corpus import parse_offsets

csv.field_size_limit(sys.maxsize)


def cmd_samples(args):
    corpus = config.corpus_dir(args.corpus_tag)
    files = sorted(corpus.glob("samples_*.jsonl"))
    if not files:
        sys.exit(f"no samples_*.jsonl under {corpus}")
    shown = 0
    for path in files:
        for line in open(path):
            if shown >= args.n:
                return
            s = json.loads(line)
            old = s["old"]
            print(f"--- doc {s['doc_id']} ({s['title']!r})  {s['qid']}")
            print(f"    {s['replaced']!r}  ->  {s['with']!r}")
            print(f"    ...{old}...".replace("\n", " "))
            print()
            shown += 1


def iter_csv(path, limit=None):
    with open(path, newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if limit is not None and i >= limit:
                return
            yield row


def cmd_verify(args):
    part_name = config.PART_TEMPLATE.format(part=args.part)
    out_csv = config.corpus_metadata_dir(args.corpus_tag) / f"{part_name}.csv"
    out_npy = config.corpus_tokenized_dir(args.corpus_tag) / f"{part_name}.npy"
    src_csv = config.SRC_METADATA_DIR / f"{part_name}.csv"
    src_npy = config.SRC_TOKENIZED_DIR / f"{part_name}.npy"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.TOKENIZER_ID, subfolder=config.TOKENIZER_SUBFOLDER, use_fast=True)

    new_mm = np.memmap(out_npy, mode="r", dtype=np.uint32)
    src_mm = np.memmap(src_npy, mode="r", dtype=np.uint32)

    n = dict(docs=0, rewritten=0, passthrough=0, mention_checked=0,
             mention_span_bad=0, tok_span_bad=0, eos_bad=0, contiguity_bad=0)
    expected_start = 0
    src_iter = iter_csv(src_csv, limit=args.n_docs)

    for row in iter_csv(out_csv, limit=args.n_docs):
        n["docs"] += 1
        start, end = int(row[0]), int(row[1])
        if start != expected_start:
            n["contiguity_bad"] += 1
        expected_start = end
        tokens = new_mm[start:end]
        if len(tokens) == 0 or tokens[-1] != config.EOS_TOKEN_ID:
            n["eos_bad"] += 1
        entities = json.loads(row[6])
        offsets = parse_offsets(row[7])
        text = tokenizer.decode(tokens[:-1], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)

        src_row = next(src_iter, None)
        if src_row is not None and src_row[2] == row[2]:
            src_tokens = src_mm[int(src_row[0]):int(src_row[1])]
            if len(src_tokens) == len(tokens) and bool(np.array_equal(src_tokens, tokens)):
                n["passthrough"] += 1
            else:
                n["rewritten"] += 1

        for m in entities:
            n["mention_checked"] += 1
            cs, ce = m["char_start"], m["char_end"]
            if text[cs:ce] != m["text_mention"]:
                n["mention_span_bad"] += 1
                if args.verbose:
                    print(f"BAD SPAN doc={row[2]} {m['text_mention']!r} != "
                          f"{text[cs:ce]!r}")
                continue
            ts, te = m.get("tok_start"), m.get("tok_end")
            if ts is None or te is None or te > len(offsets):
                n["tok_span_bad"] += 1
                continue
            # The token span must cover the char span.
            if not (offsets[ts][0] <= cs and ce <= offsets[te - 1][1]):
                n["tok_span_bad"] += 1

    # The final doc's `end` must equal the npy length (only meaningful when
    # verifying the whole part, i.e. without --n-docs).
    if args.n_docs is None and expected_start != len(new_mm):
        print(f"WARNING: last end={expected_start} != npy tokens={len(new_mm)}")

    print(json.dumps(n, indent=2))
    ok = (n["mention_span_bad"] == 0 and n["tok_span_bad"] == 0
          and n["eos_bad"] == 0 and n["contiguity_bad"] == 0)
    print("VERIFY:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def _last_csv_row(path: Path, probe: int = 1 << 20):
    """Read the final line of a huge CSV without scanning it.

    Rows can be megabytes (the `offsets` column), so walk backwards in chunks
    until a newline separating the last two rows is found.
    """
    size = path.stat().st_size
    with open(path, "rb") as f:
        end, chunk = size, probe
        while chunk <= size:
            f.seek(max(0, size - chunk))
            data = f.read(chunk if size - chunk >= 0 else size)
            body = data[:-1] if data.endswith(b"\n") else data
            nl = body.rfind(b"\n")
            if nl != -1:
                return body[nl + 1:].decode("utf-8", "replace")
            if size - chunk <= 0:
                return body.decode("utf-8", "replace")
            chunk *= 4
    return None


def cmd_integrity(args):
    """Whole-file checks over every part: nothing lost, nothing truncated.

    Cheap -- reads only file sizes, the progress files, and the final CSV row of
    each part, so it runs in seconds no matter how large the corpus is.
    """
    corpus = config.corpus_dir(args.corpus_tag)
    ok_all = True
    print(f"corpus: {corpus}\n")
    hdr = ("part", "docs done", "src docs", "npy tokens", "last row end", "csv.gz", "status")
    print("%-6s %11s %10s %14s %14s %9s  %s" % hdr)
    print("-" * 88)
    for p in config.PARTS:
        name = config.PART_TEMPLATE.format(part=p)
        prog_p = corpus / f"progress_{name}.json"
        npy = config.corpus_tokenized_dir(args.corpus_tag) / f"{name}.npy"
        out_csv = config.corpus_metadata_dir(args.corpus_tag) / f"{name}.csv"
        gz = config.corpus_tokenized_dir(args.corpus_tag) / f"{name}.csv.gz"
        src_idx = config.SRC_METADATA_DIR / f"metadata-{name}.npy"

        problems = []
        if not prog_p.exists():
            print("%-6s %11s %10s %14s %14s %9s  %s"
                  % (p, "-", "-", "-", "-", "-", "NOT STARTED"))
            ok_all = False
            continue
        prog = json.loads(prog_p.read_text())
        docs_done = prog["docs_done"]
        if not prog.get("done"):
            problems.append("still running / incomplete")

        src_docs = len(np.load(src_idx, mmap_mode="r")) if src_idx.exists() else None
        if src_docs is not None and prog.get("done") and docs_done != src_docs:
            problems.append(f"document count {docs_done} != source {src_docs}")

        npy_tokens = npy.stat().st_size // 4 if npy.exists() else None
        if npy.exists() and npy.stat().st_size % 4:
            problems.append("npy size not a multiple of 4 bytes")

        last_end = None
        if out_csv.exists():
            row = _last_csv_row(out_csv)
            try:
                last_end = int(next(csv.reader([row]))[1])
            except Exception:
                problems.append("could not parse final CSV row")
        else:
            problems.append("metadata CSV missing")

        # The strongest cheap check: the final row's token offset must land
        # exactly on the end of the token file. Any lost/duplicated/truncated
        # write shifts the running offset and breaks this.
        if last_end is not None and npy_tokens is not None and last_end != npy_tokens:
            problems.append(f"last row end {last_end} != npy tokens {npy_tokens}")

        if not gz.exists():
            problems.append("csv.gz missing (gzip step did not finish)")

        status = "OK" if not problems else "; ".join(problems)
        if problems:
            ok_all = False
        print("%-6s %11s %10s %14s %14s %9s  %s" % (
            p, f"{docs_done:,}", f"{src_docs:,}" if src_docs else "?",
            f"{npy_tokens:,}" if npy_tokens else "-",
            f"{last_end:,}" if last_end else "-",
            "yes" if gz.exists() else "NO", status))

    print()
    print("INTEGRITY:", "OK" if ok_all else "PROBLEMS FOUND")
    sys.exit(0 if ok_all else 1)


def _sentence(text: str, a: int, b: int, window: int = 260):
    """The sentence-ish span of `text` containing [a, b)."""
    lo = max(0, a - window)
    hi = min(len(text), b + window)
    left = max(text.rfind(". ", lo, a), text.rfind("\n", lo, a))
    start = left + 2 if left != -1 else lo
    right = text.find(". ", b, hi)
    nl = text.find("\n", b, hi)
    if nl != -1 and (right == -1 or nl < right):
        right = nl
    end = right + 1 if right != -1 else hi
    prefix = "" if start == 0 else "..."
    suffix = "" if end >= len(text) else "..."
    return prefix + text[start:end].strip().replace("\n", " ") + suffix, start, end


def _category(surface: str, name: str) -> str:
    from build_explicit_corpus import (DETERMINERS, OTHER_PRONOUNS,
                                       POSSESSIVE_PRONOUNS, _shares_word)
    low = surface.strip().lower()
    if low in POSSESSIVE_PRONOUNS:
        return "possessive pronoun"
    if low in OTHER_PRONOUNS:
        return "pronoun"
    words = low.split()
    determiner_led = bool(words) and words[0].strip(".,'’") in DETERMINERS
    if _shares_word(surface, name):
        return "partial name" + (" (with determiner)" if determiner_led else "")
    if low.endswith("'s") or low.endswith("’s"):
        return "possessive description"
    if determiner_led:
        return "description"
    return "other"


def cmd_review(args):
    """Show before/after in sentence context, grouped by mention type."""
    import random as _random
    from build_explicit_corpus import (apply_replacements, parse_offsets,
                                       select_replacements, Stats)
    from transformers import AutoTokenizer

    part_name = config.PART_TEMPLATE.format(part=args.part)
    src_csv = config.SRC_METADATA_DIR / f"{part_name}.csv"
    idx_path = config.SRC_METADATA_DIR / f"metadata-{part_name}.npy"
    tokenizer = AutoTokenizer.from_pretrained(
        config.TOKENIZER_ID, subfolder=config.TOKENIZER_SUBFOLDER, use_fast=True)
    tokens_mm = np.memmap(config.SRC_TOKENIZED_DIR / f"{part_name}.npy",
                          mode="r", dtype=np.uint32)

    target_qids = None
    if args.target_qids == "popqa":
        from build_explicit_corpus import load_popqa_qids
        target_qids = load_popqa_qids()

    kept = {}        # category -> list of (before, after, surface, new, qid, title)
    skipped = {}     # reason -> list of (context, surface, would_become, title)
    stats = Stats()
    n_docs = 0

    with open(src_csv, "rb") as f_in:
        if args.skip and idx_path.exists():
            arr = np.load(idx_path, mmap_mode="r")
            f_in.seek(int(arr[min(args.skip, len(arr) - 1)]))
        reader = csv.reader(io.TextIOWrapper(f_in, encoding="utf-8", newline=""))
        for row in reader:
            if n_docs >= args.n_docs:
                break
            n_docs += 1
            toks = tokens_mm[int(row[0]):int(row[1])]
            if len(toks) == 0 or toks[-1] != config.EOS_TOKEN_ID:
                continue
            entities = json.loads(row[6])
            text = tokenizer.decode(toks[:-1], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            log = []
            reps = select_replacements(
                entities, text, target_qids=target_qids,
                min_coref=args.min_coref_score, min_agg=0.0, fraction=1.0,
                rng=_random.Random(f"0:{row[2]}"), skip_pronouns=False,
                allow_multiline=False, name_guard=True,
                max_words=args.max_mention_words, stats=stats, skipped_log=log)
            for s in log:
                skipped.setdefault(s["reason"], []).append(
                    (s["context"].replace("\n", " "), s["surface"],
                     s["would_become"], row[5]))
            if not reps:
                continue
            new_text = apply_replacements(text, reps)
            delta = 0
            for r in reps:
                new_start = r["start"] + delta
                new_end = new_start + len(r["new"])
                delta += len(r["new"]) - (r["end"] - r["start"])
                name = r["new"][:-2] if r["new"].endswith("'s") else r["new"]
                cat = _category(text[r["start"]:r["end"]], name)
                before, _, _ = _sentence(text, r["start"], r["end"])
                after, _, _ = _sentence(new_text, new_start, new_end)
                kept.setdefault(cat, []).append(
                    (before, after, text[r["start"]:r["end"]], r["new"],
                     r["qid"], row[5]))

    order = ["pronoun", "possessive pronoun", "description",
             "possessive description", "partial name",
             "partial name (with determiner)", "other"]
    cats = [c for c in order if c in kept] + [c for c in kept if c not in order]
    total = sum(len(v) for v in kept.values())

    out = []
    out.append(f"# Rewrite review — part {args.part}, {n_docs} documents "
               f"(from doc {args.skip})\n")
    out.append(f"{total} replacements across {len(cats)} categories. "
               f"Each entry shows the sentence BEFORE and AFTER.\n")
    for c in cats:
        items = kept[c]
        out.append(f"\n## {c}  ({len(items)} of {total}, "
                   f"{100 * len(items) / total:.1f}%)\n")
        for before, after, surf, new, qid, title in items[:args.per_category]:
            out.append(f"- `{surf}` -> `{new}`  ({qid}, article *{title}*)")
            out.append(f"    - before: {before}")
            out.append(f"    - after:  {after}")
    if skipped:
        out.append(f"\n\n# Held back by the guards\n")
        for reason, items in sorted(skipped.items()):
            out.append(f"\n## skipped: {reason}  ({len(items)} logged)\n")
            for ctx, surf, would, title in items[:args.per_category]:
                out.append(f"- `{surf}` would have become `{would}`  "
                           f"(article *{title}*)")
                out.append(f"    - context: ...{ctx}...")

    body = "\n".join(out)
    if args.out:
        Path(args.out).write_text(body)
        print(f"wrote {args.out}  ({total} replacements, "
              f"{sum(len(v) for v in skipped.values())} skips)")
    else:
        print(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ip = sub.add_parser("integrity",
                        help="whole-file checks over all parts (seconds)")
    ip.add_argument("--corpus-tag", default="explicit_all")
    ip.set_defaults(func=cmd_integrity)

    rp = sub.add_parser("review", help="before/after sentences by mention type")
    rp.add_argument("--part", type=int, default=0, choices=config.PARTS)
    rp.add_argument("--n-docs", type=int, default=400,
                    help="documents to scan")
    rp.add_argument("--skip", type=int, default=0,
                    help="start at this document index (sample elsewhere in the part)")
    rp.add_argument("--per-category", type=int, default=8,
                    help="examples to show per category")
    rp.add_argument("--target-qids", choices=("all", "popqa"), default="all")
    rp.add_argument("--min-coref-score", type=float, default=config.MIN_COREF_SCORE)
    from build_explicit_corpus import MAX_MENTION_WORDS
    rp.add_argument("--max-mention-words", type=int, default=MAX_MENTION_WORDS)
    rp.add_argument("--out", default=None, help="write markdown here instead of stdout")
    rp.set_defaults(func=cmd_review)

    sp = sub.add_parser("samples")
    sp.add_argument("--corpus-tag", default="explicit_all")
    sp.add_argument("--n", type=int, default=20)
    sp.set_defaults(func=cmd_samples)

    vp = sub.add_parser("verify")
    vp.add_argument("--corpus-tag", default="explicit_all")
    vp.add_argument("--part", type=int, default=0, choices=config.PARTS)
    vp.add_argument("--n-docs", type=int, default=None,
                    help="verify only the first N docs (default: all)")
    vp.add_argument("--verbose", action="store_true")
    vp.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
