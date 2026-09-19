"""Rewrite implicit entity mentions in the LMEnt corpus to explicit names.

Implements the causal intervention proposed in LMEnt S6.2: replace implicit
mentions (pronouns, aliases, descriptors -- mentions whose only evidence is
coreference) with the entity's canonical name, producing a new pretraining
corpus in the exact on-disk format OLMo-core's `kas_vsl` dataset consumes:

    <out>/dataset-tokenized/part-N-00000.npy      headerless uint32 token stream
    <out>/dataset-tokenized/part-N-00000.csv.gz   per-doc metadata (for bucketing)
    <out>/dataset-cache/dataset-metadata/part-N-00000.csv   same, decompressed
                                                  (what NumpyKASVSLDataset reads)

A mention is *implicit* when its best candidate entity has zero `hyperlinks`
and zero `entity_linking` score, i.e. it was found only by coreference
resolution. For each such mention we substitute the candidate's canonical
`name` (possessive pronouns like "his"/"their" become "<Name>'s"), then
re-tokenize the document, recompute every mention's character span and token
span (`tok_start`/`tok_end`, required by OLMo-core's entity-aware chunking),
and re-emit the document. Unchanged documents are copied verbatim (tokens are
byte-copied; only their metadata gains token spans and shifted start/end).

The three experiment axes from the project plan map to flags:
  * basic approach (replace everything)   -> default (--scope all, fraction 1.0)
  * only chunks with PopQA facts          -> --scope popqa / --target-qids popqa
  * partial replacement (additive effect) -> --replace-fraction 0.5

Crash-safe: progress is checkpointed every --checkpoint-every documents and a
rerun resumes exactly where it left off (see "Resume" below), so preemption on
the killable partitions costs at most one checkpoint interval.

Run one part per job (slurm array 0-7):
    python build_explicit_corpus.py --part 0 --corpus-tag explicit_all
Resume after a failure/preemption -- same command, nothing to change:
    python build_explicit_corpus.py --part 0 --corpus-tag explicit_all
Start over from scratch:
    python build_explicit_corpus.py --part 0 --corpus-tag explicit_all --restart
Smoke test:
    python build_explicit_corpus.py --part 0 --limit 2000 --corpus-tag smoke \
        --samples 100 --log-skipped 100
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import random
import re
import shutil
import sys
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: config.py, src/
import config

csv.field_size_limit(sys.maxsize)

CSV_HEADERS = ["start", "end", "id", "src", "loc", "title", "entities", "offsets"]

# Pronoun surface forms. Possessive determiners/pronouns get "<Name>'s";
# everything else becomes the bare name.
POSSESSIVE_PRONOUNS = {"his", "her", "their", "its", "hers", "theirs"}
OTHER_PRONOUNS = {
    "he", "she", "they", "it", "him", "them", "himself", "herself",
    "themselves", "itself",
}
# First/second person pronouns are speaker-relative and nearly always sit inside
# a direct quotation: substituting a name corrupts the quote and the grammar
# ("I've had a wonderful time" -> "Carrie Keagan've had a wonderful time").
PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
}
# Relative pronouns head a subordinate clause; replacing one with a name leaves
# a comma splice ("Keagan, who departed" -> "Keagan, Carrie Keagan departed").
RELATIVE_PRONOUNS = {"who", "whom", "whose", "which", "that"}
# Descriptor-introducing words. A mention starting with one of these is a
# description ("the president", "his hometown"), never a bare proper name, so
# the name-overlap guard below does not apply to it.
DETERMINERS = {
    "the", "a", "an", "this", "that", "these", "those", "both", "either",
    "his", "her", "hers", "its", "their", "theirs", "our", "ours", "my",
    "your", "yours",
}
# Replacing a mention longer than this many words deletes surrounding content
# rather than renaming an entity, so such spans are left alone. 95.6% of
# eligible mentions are 4 words or fewer.
MAX_MENTION_WORDS = 4
# Entity-linking confidence at or above which we treat a mention as an
# authoritative statement of identity that can contradict a coref claim.
MIN_EXPLICIT_EL = 0.5


# --------------------------------------------------------------------------- #
# Fast (de)serialization of the CSV columns
# --------------------------------------------------------------------------- #
def parse_offsets(s: str) -> np.ndarray:
    """'[(0, 2), (2, 6)]' -> int64 array of shape (n_tokens, 2).

    ~30x faster than ast.literal_eval on the multi-MB offset strings.
    """
    body = s.strip()[1:-1]
    if not body:
        return np.empty((0, 2), dtype=np.int64)
    flat = np.array(body.replace("(", "").replace(")", "").split(","), dtype=np.int64)
    return flat.reshape(-1, 2)


def format_offsets(offsets) -> str:
    return "[" + ", ".join(f"({int(a)}, {int(b)})" for a, b in offsets) + "]"


def dump_entities(entities: list[dict]) -> str:
    return json.dumps(entities, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Mention classification
# --------------------------------------------------------------------------- #
def best_candidate(mention: dict) -> dict | None:
    cands = mention.get("candidates") or []
    if not cands:
        return None
    return max(cands, key=lambda c: c.get("aggregated_score", 0.0))


def is_implicit(cand: dict) -> bool:
    s = cand.get("scores_by_source", {})
    return (
        s.get("hyperlinks", 0.0) == 0.0
        and s.get("entity_linking", 0.0) == 0.0
        and max(s.get("coref", 0.0), s.get("coref_cluster", 0.0)) > 0.0
    )


SENTENCE_BREAK = re.compile(r'(?:[.!?]["\')\]]?\s+|\n+)')


def sentence_starts(text: str) -> list[int]:
    """Character offsets where each sentence begins."""
    return [0] + [m.end() for m in SENTENCE_BREAK.finditer(text)]


def _sentence_index(starts: list[int], pos: int) -> int:
    return bisect_right(starts, pos) - 1


def _words(s: str) -> list[str]:
    """Lowercased alphanumeric words, punctuation-insensitive."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def _name_already_in_context(name: str, context: str) -> bool:
    """True if every word of the canonical name is already in the local context.

    Guards the partial-name case: the surface is a fragment of the name and the
    surrounding words supply the rest, so substituting the full name duplicates
    text -- "in the 1953 Big Ten Conference football season" would become
    "in the 1953 1953 Big Ten Conference football season football season".
    """
    nw = set(_words(name))
    return bool(nw) and nw <= set(_words(context))


def _contains_full_name(surface: str, name: str) -> bool:
    """True if the surface already spells out every word of the canonical name.

    Then the mention is *already explicit* -- coreference merely wrapped the
    name in a determiner or a longer phrase ("the Shellbend", '"John Minturn"',
    "the New Jersey Senate"). Rewriting adds no entity information and only
    deletes the surrounding words, so these are left alone.
    """
    nw = _words(name)
    return bool(nw) and set(nw) <= set(_words(surface))


def _is_name_like(surface: str) -> bool:
    """True if the surface reads as a bare proper name rather than a description.

    Descriptions carry a determiner or possessive ("the president", "his
    hometown"), or start lowercase ("president"); names do not ("California",
    "Queen City"). Only names are subject to the overlap guard below -- so
    "the president" -> "Donald Trump" is allowed even with no shared word,
    while "California" -> "Los Angeles" is not.
    """
    words = surface.strip().split()
    if not words:
        return False
    if words[0].strip(".,'’").lower() in DETERMINERS:
        return False                      # a description, not a name
    first = words[0]
    return len(first) >= 3 and first[0].isupper() and not first.isupper()


def _lexically_related(surface: str, name: str) -> bool:
    """Is the surface lexically connected to the canonical name at all?

    Word overlap, a shared word stem ("Ron" / "Ronald Taylor"), or a close
    spelling ("Toolsboro" / "Toolesboro") all count. Used only to decide
    whether a name-like surface is plausibly a short form of the target, so
    punctuation and case must not matter -- an earlier version compared raw
    whitespace tokens and wrongly blocked "Baumhammers\'" / "Richard
    Baumhammers".
    """
    sw = [w for w in _words(surface) if len(w) >= 3 and w not in DETERMINERS]
    nw = [w for w in _words(name) if len(w) >= 3 and w not in DETERMINERS]
    if not sw or not nw:
        return False
    if set(sw) & set(nw):
        return True
    for a in sw:                       # shared stem, e.g. Ron / Ronald
        for b in nw:
            if a.startswith(b) or b.startswith(a) or a in b or b in a:
                return True
    from difflib import SequenceMatcher   # near-identical spelling
    return SequenceMatcher(None, " ".join(sw), " ".join(nw)).ratio() >= 0.8


def replacement_text(surface: str, name: str, *, skip_pronouns: bool,
                     allow_multiline: bool = False,
                     name_guard: bool = True,
                     max_words: int = MAX_MENTION_WORDS,
                     context: str | None = None,
                     surface_confirmed: bool = False) -> tuple[str | None, str]:
    """The explicit string to substitute, plus a reason when we decline.

    Returns (replacement, "") on success or (None, reason) when the mention is
    left alone; the reason is what --log-skipped records.
    """
    stripped = surface.strip()
    if not stripped or not name:
        return None, "empty"
    if not allow_multiline and "\n" in surface:
        # Coref spans crossing line breaks are usually the title + first body
        # mention glued together; replacing them restructures the document.
        return None, "multiline"
    low = stripped.lower()
    if low in PERSON_PRONOUNS:
        return None, "person_pronoun"
    if low in RELATIVE_PRONOUNS:
        return None, "relative_pronoun"
    if not any(ch.isupper() for ch in name):
        # The "canonical name" is a common noun ("post office", "folding
        # boat"), so substituting it neither makes a name explicit nor reads
        # grammatically ("until they moved" -> "until post office moved").
        return None, "name_not_proper"
    # Pronouns are otherwise safe: one word, never containing the name.
    if low in POSSESSIVE_PRONOUNS:
        return (None, "pronoun") if skip_pronouns else (name + "'s", "")
    if low in OTHER_PRONOUNS:
        return (None, "pronoun") if skip_pronouns else (name, "")
    if max_words and len(_words(stripped)) > max_words:
        # Long coref spans are appositions and relative clauses that merely
        # *contain* the referent ("Greene, who had been committed to play
        # college baseball for ...", "her mother Winifred (nee Reid)").
        # Substituting the name for the whole span deletes real content.
        return None, "too_long"
    if _contains_full_name(stripped, name):
        return None, "contains_name"      # already explicit -- nothing to add
    if (context is not None and set(_words(stripped)) <= set(_words(name))
            and _name_already_in_context(name, context)):
        return None, "name_in_context"    # the context already spells it out
    # A possessive surface must stay possessive: the annotation's span often
    # includes the clitic ("the boat's", "the City's"), and dropping it yields
    # "a fifth of Shellbend original size".
    possessive = low.endswith("'s") or low.endswith("’s")
    already_possessive = name.rstrip().lower().endswith(("'s", "’s", "s'"))
    target = name + "'s" if possessive and not already_possessive else name
    if (name_guard and not surface_confirmed and _is_name_like(stripped)
            and not _lexically_related(stripped, name)):
        # A name-like surface with zero lexical overlap with the target name is
        # usually a coref-cluster misattribution (e.g. "California" -> "Los
        # Angeles"). Descriptions ("the president") are exempt -- see
        # _is_name_like -- so real indirect mentions still get rewritten.
        # `surface_confirmed` exempts aliases a stronger source vouched for in
        # this same document ("Rini" -> "Chibiusa", "Leningrad" -> "Saint
        # Petersburg"): those cannot be misattributions.
        return None, "name_guard"
    return target, ""


# --------------------------------------------------------------------------- #
# Span remapping after replacement
# --------------------------------------------------------------------------- #
def remap_entities(entities, replacements, new_text):
    """Recompute char spans of all mentions after applying `replacements`.

    `replacements`: sorted, non-overlapping list of dicts with keys
    idx (mention index), start, end (old span), new (replacement string).

    Returns the new entities list. Mentions whose boundary falls strictly
    inside a replaced region, or that lie inside one, are dropped (their text
    no longer exists); mentions that *contain* replaced regions survive with
    an updated `text_mention`.
    """
    starts = [r["start"] for r in replacements]
    ends = [r["end"] for r in replacements]
    # Cumulative length delta of all replacements ending at or before a position.
    deltas = [len(r["new"]) - (r["end"] - r["start"]) for r in replacements]
    cum = np.cumsum([0] + deltas)
    replaced_by_idx = {r["idx"]: r for r in replacements}

    def shift(pos: int) -> int:
        # Sum of deltas for replacements with end <= pos.
        return int(cum[bisect_right(ends, pos)])

    def crosses(pos: int) -> bool:
        # True if pos falls strictly inside some replaced region.
        i = bisect_right(starts, pos) - 1
        return i >= 0 and starts[i] < pos < ends[i]

    out, dropped = [], 0
    for idx, m in enumerate(entities):
        cs, ce = m["char_start"], m["char_end"]
        if idx in replaced_by_idx:
            r = replaced_by_idx[idx]
            new_cs = r["start"] + shift(r["start"])
            new_m = dict(m)
            new_m["char_start"] = new_cs
            new_m["char_end"] = new_cs + len(r["new"])
            new_m["rewritten_from"] = m["text_mention"]
            new_m["text_mention"] = r["new"]
            out.append(new_m)
            continue
        if crosses(cs) or crosses(ce):
            dropped += 1                  # boundary inside a replaced region
            continue
        # Contained in a replaced region (duplicate/nested span of the same text)?
        i = bisect_right(starts, cs) - 1
        if i >= 0 and starts[i] <= cs and ce <= ends[i]:
            dropped += 1
            continue
        new_cs, new_ce = cs + shift(cs), ce + shift(ce)
        new_m = dict(m)
        new_m["char_start"], new_m["char_end"] = new_cs, new_ce
        seg = new_text[new_cs:new_ce]
        if seg != m["text_mention"]:
            # The mention contains a replaced region -- its surface changed.
            new_m["text_mention"] = seg
        out.append(new_m)
    return out, dropped


def add_token_spans(entities, offsets: np.ndarray, text: str | None = None):
    """Attach tok_start/tok_end (needed by OLMo-core's entity-aware chunking).

    Mirrors dolma's `add_token_indices`: tok_start is the token whose span
    contains char_start, tok_end is one past the token whose span contains
    char_end. Mentions that can't be located are dropped, matching upstream.
    When `text` is given, mentions whose span doesn't reproduce their
    text_mention are dropped too (extra safety on rewritten docs).
    """
    if len(offsets) == 0:
        return [], len(entities)
    tok_starts, tok_ends = offsets[:, 0], offsets[:, 1]
    out, dropped = [], 0
    for m in entities:
        cs, ce = m["char_start"], m["char_end"]
        # first token with end > cs
        i = int(np.searchsorted(tok_ends, cs, side="right"))
        # first token with end >= ce
        j = int(np.searchsorted(tok_ends, ce, side="left"))
        if (
            i >= len(offsets) or j >= len(offsets)
            or not (tok_starts[i] <= cs < tok_ends[i])
            or not (tok_starts[j] < ce <= tok_ends[j])
            or (text is not None and text[cs:ce] != m["text_mention"])
        ):
            dropped += 1
            continue
        m["tok_start"], m["tok_end"] = i, j + 1
        out.append(m)
    return out, dropped


# --------------------------------------------------------------------------- #
# Stats / progress
# --------------------------------------------------------------------------- #
STAT_KEYS = [
    "docs_total", "docs_rewritten", "docs_passthrough", "docs_decode_mismatch",
    "docs_skipped_scope", "mentions_total", "mentions_implicit",
    "mentions_replaced", "mentions_replaced_pronoun",
    "mentions_dropped_overlap", "mentions_dropped_tokenspan", "tokens_in",
    "tokens_out",
]
SKIP_KEYS = ["skip_multiline", "skip_name_guard", "skip_pronoun",
             "skip_person_pronoun", "skip_relative_pronoun",
             "skip_name_not_proper", "skip_contains_name", "skip_name_in_context",
             "skip_too_long", "skip_empty", "skip_span_invalid",
             "skip_not_targeted", "skip_low_score", "skip_fraction",
             "skip_sentence_has_explicit", "skip_sentence_duplicate",
             "skip_contradicted", "name_guard_waived"]


class Stats:
    def __init__(self, initial: dict | None = None):
        self.d = {k: 0 for k in STAT_KEYS + SKIP_KEYS}
        if initial:
            self.d.update({k: v for k, v in initial.items() if k in self.d})

    def __getitem__(self, k):
        return self.d[k]

    def add(self, k, n=1):
        self.d[k] = self.d.get(k, 0) + n


def input_row_offset(part_name: str, row_idx: int) -> int | None:
    """Byte offset of input CSV line `row_idx`, via the authors' line index."""
    idx_path = config.SRC_METADATA_DIR / f"metadata-{part_name}.npy"
    if not idx_path.exists():
        return None
    arr = np.load(idx_path, mmap_mode="r")
    if row_idx >= len(arr):
        return None
    return int(arr[row_idx])


# --------------------------------------------------------------------------- #
# Main per-part pipeline
# --------------------------------------------------------------------------- #
def load_popqa_qids() -> set[str]:
    from datasets import load_dataset
    ds = load_dataset(config.POPQA_DATASET_ID, split="train")
    qids = set()
    for col in ("s_uri", "o_uri"):
        qids.update(q for q in ds[col] if q)
    return qids


def explicit_assertions(entities):
    """What hyperlinks / confident entity linking claim, indexed for lookup.

    Returns (by_span, by_surface): span or surface -> set of QIDs asserted by a
    source stronger than coreference. Used to veto coref claims that contradict
    them -- coref calling a span "France" a mention of "Prime Minister of
    France" is wrong when entity linking says that span is Q142.
    """
    by_span, by_surface = {}, {}
    for m in entities:
        for c in (m.get("candidates") or []):
            sc = c.get("scores_by_source", {})
            if (sc.get("hyperlinks", 0.0) > 0.0
                    or sc.get("entity_linking", 0.0) >= MIN_EXPLICIT_EL):
                by_span.setdefault((m["char_start"], m["char_end"]), set()).add(c["qid"])
                by_surface.setdefault(m["text_mention"].strip().lower(), set()).add(c["qid"])
    return by_span, by_surface


def select_replacements(entities, text, *, target_qids, min_coref, min_agg,
                        fraction, rng, skip_pronouns, allow_multiline,
                        name_guard, max_words, stats, skipped_log=None,
                        once_per_sentence=True):
    """Decide which mentions to rewrite; returns sorted non-overlapping list."""
    starts = sentence_starts(text) if once_per_sentence else None
    # Sentences where an entity is ALREADY named explicitly (hyperlink or
    # entity-linking evidence). An implicit mention there needs no rewriting:
    # the fact is already learnable in that sentence, and replacing the pronoun
    # only yields "Keagan announced Carrie Keagan's decision".
    by_span, by_surface = explicit_assertions(entities)
    explicit_in_sentence = set()
    if once_per_sentence:
        for m in entities:
            c = best_candidate(m)
            if c is not None and not is_implicit(c):
                explicit_in_sentence.add(
                    (_sentence_index(starts, m["char_start"]), c.get("qid")))
    picked = []
    for idx, m in enumerate(entities):
        stats.add("mentions_total")
        cand = best_candidate(m)
        if cand is None or not is_implicit(cand):
            continue
        s = cand.get("scores_by_source", {})
        if max(s.get("coref", 0.0), s.get("coref_cluster", 0.0)) < min_coref:
            stats.add("skip_low_score")
            continue
        if cand.get("aggregated_score", 0.0) < min_agg:
            stats.add("skip_low_score")
            continue
        stats.add("mentions_implicit")
        if target_qids is not None and cand.get("qid") not in target_qids:
            stats.add("skip_not_targeted")
            continue
        cs, ce = m["char_start"], m["char_end"]
        if not (0 <= cs < ce <= len(text)) or text[cs:ce] != m["text_mention"]:
            stats.add("skip_span_invalid")  # span doesn't validate -- never touch it
            continue
        qid = cand.get("qid")
        span_says = by_span.get((cs, ce))
        surf_says = by_surface.get(m["text_mention"].strip().lower())
        # Does a stronger source AGREE that this surface names this entity
        # somewhere in this document? Then it is a vouched-for alias.
        surface_confirmed = qid in (span_says or set()) or qid in (surf_says or set())
        if surface_confirmed:
            stats.add("name_guard_waived")
        if (span_says and qid not in span_says) or (surf_says and qid not in surf_says):
            # A stronger source names this span/surface as a different entity.
            stats.add("skip_contradicted")
            if skipped_log is not None:
                skipped_log.append({
                    "reason": "contradicted", "surface": m["text_mention"],
                    "would_become": cand.get("name", ""), "qid": qid,
                    "context": text[max(0, cs - 70):ce + 70]})
            continue
        name = cand.get("name", "")
        pad = len(name) + 24
        new, reason = replacement_text(m["text_mention"], name,
                                       skip_pronouns=skip_pronouns,
                                       allow_multiline=allow_multiline,
                                       name_guard=name_guard,
                                       max_words=max_words,
                                       context=text[max(0, cs - pad):ce + pad],
                                       surface_confirmed=surface_confirmed)
        if new is None:
            stats.add(f"skip_{reason}")
            if skipped_log is not None and reason in (
                    "name_guard", "multiline", "contains_name", "too_long",
                    "name_in_context", "name_not_proper", "person_pronoun",
                    "relative_pronoun"):
                skipped_log.append({"reason": reason, "surface": m["text_mention"],
                                    "would_become": name, "qid": cand.get("qid"),
                                    "context": text[max(0, cs - 70): ce + 70]})
            continue
        if fraction < 1.0 and rng.random() >= fraction:
            stats.add("skip_fraction")
            continue
        picked.append({"idx": idx, "start": cs, "end": ce, "new": new,
                       "qid": cand.get("qid"),
                       "score": cand.get("aggregated_score", 0.0),
                       "pronoun": m["text_mention"].strip().lower()
                                  in (POSSESSIVE_PRONOUNS | OTHER_PRONOUNS)})
    # Greedy pass in reading order: resolve overlaps (earlier start wins, ties
    # to the higher-scoring candidate) and enforce one explicit name per entity
    # per sentence.
    picked.sort(key=lambda r: (r["start"], -r["score"]))
    kept, last_end, claimed = [], -1, set()
    for r in picked:
        if r["start"] < last_end:
            stats.add("mentions_dropped_overlap")
            continue
        if once_per_sentence:
            key = (_sentence_index(starts, r["start"]), r["qid"])
            if key in explicit_in_sentence:
                stats.add("skip_sentence_has_explicit")
                continue
            if key in claimed:
                # A previous mention of this entity in this sentence was already
                # made explicit; a second one is pure repetition ("Ricardo
                # Moyano has a channel on which Ricardo Moyano uploads").
                stats.add("skip_sentence_duplicate")
                continue
            claimed.add(key)
        kept.append(r)
        last_end = r["end"]
    return kept


def apply_replacements(text, replacements) -> str:
    pieces, pos = [], 0
    for r in replacements:
        pieces.append(text[pos:r["start"]])
        pieces.append(r["new"])
        pos = r["end"]
    pieces.append(text[pos:])
    return "".join(pieces)


def process_part(args):
    part_name = config.PART_TEMPLATE.format(part=args.part)
    src_npy = config.SRC_TOKENIZED_DIR / f"{part_name}.npy"
    src_csv = config.SRC_METADATA_DIR / f"{part_name}.csv"

    out_dir = config.corpus_dir(args.corpus_tag)
    out_tok_dir = config.corpus_tokenized_dir(args.corpus_tag)
    out_meta_dir = config.corpus_metadata_dir(args.corpus_tag)
    out_tok_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_tok_dir / f"{part_name}.npy"
    out_csv = out_meta_dir / f"{part_name}.csv"
    out_csv_gz = out_tok_dir / f"{part_name}.csv.gz"
    progress_path = out_dir / f"progress_{part_name}.json"
    stats_path = out_dir / f"stats_{part_name}.json"
    samples_path = out_dir / f"samples_{part_name}.jsonl"
    skipped_path = out_dir / f"skipped_{part_name}.jsonl"

    # ---------------- resume state ----------------
    prog = None
    if progress_path.exists() and not args.restart:
        prog = json.loads(progress_path.read_text())
        # Checked before the done-check too: silently handing back a corpus
        # built with other settings is worse than failing loudly.
        if prog.get("args_fingerprint") != args_fingerprint(args):
            sys.exit(
                f"[part {args.part}] {progress_path.name} was written with "
                f"different rewrite settings:\n"
                f"  saved:   {prog.get('args_fingerprint')}\n"
                f"  current: {args_fingerprint(args)}\n"
                f"Use a different --corpus-tag for the new settings, or pass "
                f"--restart to rebuild this part in place.")
        if prog.get("done"):
            print(f"[part {args.part}] already complete "
                  f"({prog['docs_done']} docs). Use --restart to redo it.")
            if not out_csv_gz.exists():
                finalize_gzip(out_csv, out_csv_gz, args.part)
            return
        print(f"[part {args.part}] RESUMING from doc {prog['docs_done']} "
              f"({prog['npy_bytes']} npy bytes, {prog['csv_bytes']} csv bytes)")

    print(f"[part {args.part}] src tokens: {src_npy}")
    print(f"[part {args.part}] src metadata: {src_csv}")
    print(f"[part {args.part}] out corpus: {out_dir}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.TOKENIZER_ID, subfolder=config.TOKENIZER_SUBFOLDER, use_fast=True)
    eos = config.EOS_TOKEN_ID

    target_qids = None
    scope_qids = None
    if args.target_qids == "popqa" or args.scope == "popqa":
        popqa_qids = load_popqa_qids()
        print(f"[popqa] {len(popqa_qids)} target QIDs")
        if args.target_qids == "popqa":
            target_qids = popqa_qids
        if args.scope == "popqa":
            scope_qids = popqa_qids

    tokens_mm = np.memmap(src_npy, mode="r", dtype=np.uint32)
    stats = Stats(prog["stats"] if prog else None)
    qid_counts: dict[str, int] = dict(prog["qid_counts"]) if prog else {}
    n_samples_written = prog["n_samples"] if prog else 0
    n_skipped_written = prog["n_skipped"] if prog else 0
    written_tokens = prog["written_tokens"] if prog else 0
    docs_done = prog["docs_done"] if prog else 0
    t0 = time.time()
    docs_at_start = docs_done

    # Truncate outputs back to the last checkpointed byte, then append. Both
    # handles are binary so the recorded offsets are exact byte positions
    # (text-mode tell() returns an opaque cookie, which truncate() can't use).
    f_npy = open(out_npy, "r+b" if prog else "wb")
    f_csv_raw = open(out_csv, "r+b" if prog else "wb")
    if prog:
        f_npy.truncate(prog["npy_bytes"])
        f_npy.seek(prog["npy_bytes"])
        f_csv_raw.truncate(prog["csv_bytes"])
        f_csv_raw.seek(prog["csv_bytes"])
    f_csv = io.TextIOWrapper(f_csv_raw, encoding="utf-8", newline="",
                             write_through=True)
    f_samples = open(samples_path, "a" if prog else "w")
    f_skipped = open(skipped_path, "a" if prog else "w")

    writer = csv.writer(f_csv)

    def emit(row, tokens: np.ndarray, ents, offsets_str: str):
        nonlocal written_tokens
        f_npy.write(np.ascontiguousarray(tokens, dtype=config.TOKEN_DTYPE).tobytes())
        writer.writerow([
            written_tokens, written_tokens + len(tokens),
            row[2], row[3], row[4], row[5],
            dump_entities(ents), offsets_str,
        ])
        written_tokens += len(tokens)
        stats.add("tokens_out", len(tokens))

    def emit_passthrough(row, doc_tokens, entities, reason: str | None = None):
        # Even untouched docs need tok_start/tok_end for OLMo-core's
        # entity-aware chunking; compute them from the stored offsets
        # (spans were already validated at annotation time, so no text check).
        offsets = parse_offsets(row[7])
        ents, dropped = add_token_spans(entities, offsets, text=None)
        stats.add("mentions_dropped_tokenspan", dropped)
        emit(row, doc_tokens, ents, row[7])
        stats.add(reason or "docs_passthrough")

    def checkpoint(done: bool = False):
        """Flush both outputs, then atomically record the consistent offsets."""
        f_csv.flush()
        os.fsync(f_csv_raw.fileno())
        f_npy.flush()
        os.fsync(f_npy.fileno())
        f_samples.flush()
        f_skipped.flush()
        state = {
            "done": done,
            "docs_done": docs_done,
            "npy_bytes": f_npy.tell(),
            "csv_bytes": f_csv_raw.tell(),
            "written_tokens": written_tokens,
            "n_samples": n_samples_written,
            "n_skipped": n_skipped_written,
            "args_fingerprint": args_fingerprint(args),
            "stats": stats.d,
            "qid_counts": qid_counts,
        }
        tmp = progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, progress_path)

    try:
        # Binary handle so a resume can seek straight to the byte offset of the
        # next input row (from the authors' per-part line index).
        with open(src_csv, "rb") as f_in_raw:
            if docs_done:
                off = input_row_offset(part_name, docs_done)
                if off is not None:
                    f_in_raw.seek(off)
                else:
                    print(f"[part {args.part}] no line index; skipping "
                          f"{docs_done} rows by scan...", flush=True)
                    for _ in range(docs_done):
                        f_in_raw.readline()
            reader = csv.reader(io.TextIOWrapper(f_in_raw, encoding="utf-8",
                                                 newline=""))

            for row in reader:
                if args.limit and docs_done >= args.limit:
                    break
                docs_done += 1
                stats.add("docs_total")
                start, end = int(row[0]), int(row[1])
                doc_tokens = tokens_mm[start:end]
                stats.add("tokens_in", len(doc_tokens))
                entities = json.loads(row[6])

                # Scope: only touch docs that mention a PopQA entity at all.
                if scope_qids is not None:
                    doc_qids = {c.get("qid") for m in entities
                                for c in (m.get("candidates") or [])}
                    if not (doc_qids & scope_qids):
                        emit_passthrough(row, doc_tokens, entities, "docs_skipped_scope")
                        continue

                if len(doc_tokens) == 0 or doc_tokens[-1] != eos:
                    emit_passthrough(row, doc_tokens, entities, "docs_decode_mismatch")
                    continue

                text = tokenizer.decode(doc_tokens[:-1], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)

                rng = random.Random(f"{args.seed}:{row[2]}")
                skipped_log = [] if n_skipped_written < args.log_skipped else None
                replacements = select_replacements(
                    entities, text,
                    target_qids=target_qids,
                    min_coref=args.min_coref_score,
                    min_agg=args.min_aggregated_score,
                    fraction=args.replace_fraction,
                    rng=rng, skip_pronouns=args.no_pronouns,
                    allow_multiline=args.allow_multiline,
                    name_guard=not args.no_name_guard,
                    max_words=args.max_mention_words,
                    once_per_sentence=not args.allow_repeat_in_sentence,
                    stats=stats, skipped_log=skipped_log)

                if skipped_log:
                    for s in skipped_log[: args.log_skipped - n_skipped_written]:
                        s["doc_id"], s["title"] = row[2], row[5]
                        f_skipped.write(json.dumps(s, ensure_ascii=False) + "\n")
                        n_skipped_written += 1

                if not replacements:
                    emit_passthrough(row, doc_tokens, entities)
                    continue

                new_text = apply_replacements(text, replacements)
                new_entities, dropped = remap_entities(entities, replacements, new_text)
                stats.add("mentions_dropped_overlap", dropped)

                enc = tokenizer(new_text, add_special_tokens=False,
                                return_offsets_mapping=True)
                new_tokens = np.array(enc["input_ids"] + [eos], dtype=np.uint32)
                offsets = np.array(enc["offset_mapping"], dtype=np.int64)
                new_entities, dropped = add_token_spans(new_entities, offsets, new_text)
                stats.add("mentions_dropped_tokenspan", dropped)

                emit(row, new_tokens, new_entities, format_offsets(offsets))
                stats.add("docs_rewritten")
                stats.add("mentions_replaced", len(replacements))
                stats.add("mentions_replaced_pronoun",
                          sum(1 for r in replacements if r["pronoun"]))
                for r in replacements:
                    qid_counts[r["qid"]] = qid_counts.get(r["qid"], 0) + 1

                if n_samples_written < args.samples:
                    for r in replacements[: args.samples - n_samples_written]:
                        ctx = 80
                        f_samples.write(json.dumps({
                            "doc_id": row[2], "title": row[5], "qid": r["qid"],
                            "old": text[max(0, r["start"] - ctx): r["end"] + ctx],
                            "replaced": text[r["start"]:r["end"]],
                            "with": r["new"],
                        }, ensure_ascii=False) + "\n")
                        n_samples_written += 1

                if docs_done % args.checkpoint_every == 0:
                    checkpoint()
                    rate = (docs_done - docs_at_start) / max(time.time() - t0, 1e-9)
                    print(f"[part {args.part}] {docs_done} docs "
                          f"({rate:.0f} docs/s), "
                          f"{stats['mentions_replaced']} replaced [checkpointed]",
                          flush=True)

        checkpoint(done=True)
    finally:
        f_npy.close()
        f_csv.close()
        f_samples.close()
        f_skipped.close()

    finalize_gzip(out_csv, out_csv_gz, args.part)

    final = dict(stats.d)
    final["top_qids"] = sorted(qid_counts.items(), key=lambda kv: -kv[1])[:50]
    final["args"] = {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()}
    stats_path.write_text(json.dumps(final, indent=2))
    print(f"[part {args.part}] done in {(time.time() - t0) / 60:.1f} min")
    print(json.dumps({k: v for k, v in stats.d.items()}, indent=2))


def finalize_gzip(out_csv: Path, out_csv_gz: Path, part: int):
    """Gzip the metadata next to the npy (OLMo-core's bucketing reads it).

    Written to a temp file and replaced so an interrupted gzip never leaves a
    truncated .csv.gz behind.
    """
    print(f"[part {part}] gzipping metadata -> {out_csv_gz}", flush=True)
    tmp = out_csv_gz.with_suffix(".gz.tmp")
    with open(out_csv, "rb") as f_in, gzip.open(tmp, "wb", compresslevel=4) as f_out:
        shutil.copyfileobj(f_in, f_out, length=1 << 24)
    os.replace(tmp, out_csv_gz)


def args_fingerprint(args) -> str:
    """Identity of the rewrite configuration -- resuming with different
    settings would splice two different corpora together, so we refuse."""
    keys = ["part", "scope", "target_qids", "replace_fraction", "min_coref_score",
            "min_aggregated_score", "no_name_guard", "max_mention_words",
            "allow_multiline", "no_pronouns", "allow_repeat_in_sentence",
            "seed", "limit"]
    return json.dumps({k: getattr(args, k) for k in keys}, sort_keys=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=int, required=True, choices=config.PARTS,
                    help="which corpus part to process (slurm array index)")
    ap.add_argument("--corpus-tag", default="explicit_all",
                    help="output corpus name under corpora/ (one tag per "
                         "rewrite configuration)")
    ap.add_argument("--scope", choices=("all", "popqa"), default="all",
                    help="'popqa': only rewrite documents that mention a PopQA "
                         "subject/object entity; other docs pass through unchanged")
    ap.add_argument("--target-qids", choices=("all", "popqa"), default="all",
                    help="'popqa': only replace implicit mentions OF PopQA "
                         "subject/object entities (other entities' implicit "
                         "mentions are left alone)")
    ap.add_argument("--replace-fraction", type=float, default=1.0,
                    help="replace only this fraction of eligible implicit "
                         "mentions (deterministic per --seed); for the "
                         "additive-effect experiment")
    ap.add_argument("--min-coref-score", type=float, default=config.MIN_COREF_SCORE,
                    help="minimum max(coref, coref_cluster) score to trust an "
                         "implicit mention (paper's retrieval uses 0.6)")
    ap.add_argument("--min-aggregated-score", type=float,
                    default=config.MIN_AGGREGATED_SCORE)
    ap.add_argument("--allow-repeat-in-sentence", action="store_true",
                    help="rewrite EVERY implicit mention, even when the entity "
                         "is already named in the same sentence. Off by "
                         "default: 49%% of candidate rewrites share a sentence "
                         "with another mention of the same entity, where the "
                         "rewrite adds no explicitness and reads badly "
                         "(\"Keagan announced Carrie Keagan's decision\").")
    ap.add_argument("--max-mention-words", type=int, default=MAX_MENTION_WORDS,
                    help="never replace a mention longer than this many words "
                         f"(default {MAX_MENTION_WORDS}); longer coref spans are "
                         "appositions/relative clauses whose replacement would "
                         "delete real content. 0 = no limit.")
    ap.add_argument("--no-name-guard", action="store_true",
                    help="also replace bare name-like surfaces that share no "
                         "word with the target name (off by default: those are "
                         "usually coref misattributions like California->Los "
                         "Angeles). Descriptions such as 'the president' are "
                         "never affected by the guard.")
    ap.add_argument("--allow-multiline", action="store_true",
                    help="also replace mentions whose span crosses a line "
                         "break (off by default: those are usually the title "
                         "glued to the first body mention)")
    ap.add_argument("--no-pronouns", action="store_true",
                    help="only rewrite non-pronoun implicit mentions (aliases/"
                         "descriptors like 'the team'), leave he/his/their alone")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N documents (smoke test)")
    ap.add_argument("--samples", type=int, default=200,
                    help="write up to N before/after snippets to samples_*.jsonl")
    ap.add_argument("--log-skipped", type=int, default=200,
                    help="write up to N guard-skipped mentions to "
                         "skipped_*.jsonl, for auditing the name guard")
    ap.add_argument("--checkpoint-every", type=int, default=20000,
                    help="flush outputs and record resumable progress every N "
                         "documents")
    ap.add_argument("--restart", action="store_true",
                    help="ignore any saved progress and rebuild this part")
    args = ap.parse_args()
    process_part(args)


if __name__ == "__main__":
    main()
