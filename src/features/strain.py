"""Group 4 - "What else was competing for these tokens".

The LMEnt paper (§6.1) finds fact frequency correlates with *both* learning and
forgetting and says frequency "does not dictate learning and forgetting
dynamics, and may be confounded by other factors."  This module builds one
candidate confounder: **token interference**.  A fact is hard to retain when the
tokens it is made of are shared with many *other entities* the model saw nearby
in training.

Three feature families, all computable from ``popqa-kas`` alone (the "Stage 1"
universe: 19,384 QIDs).  Each has a static form and a windowed form:

  strain      how many other ENTITIES share a token with this fact
  ownership   what share of the evidence carrying this token belongs to this fact
  alt-answer  how often the subject co-occurs with a *different* valid answer
              for the same relation (the paper's own §6.3 counterfactual notion)

Three things were established empirically before this module was written and are
baked into its defaults -- see the plan doc for the numbers:

1.  **Count entities, not strings.**  Two distinct entities written "Buffalo"
    (the city, the team) are two competitors, not one; and an entity's own
    aliases are not competitors.  The unit is the QID (``s_uri`` / ``o_uri``),
    never the surface string.

2.  **Do not pool Wikidata aliases.**  Every QID in popqa-kas appears with
    exactly one observed surface form, while ``s_aliases``/``o_aliases`` add
    2.1x more forms (up to 208 for one entity).  Pooling them inflates median
    object strain several-fold -- 11 -> 56 per fact over all of popqa-kas, 4 -> 75
    on the top-decile panel -- and destroys the signal, because an alias list
    is a knowledge-base artifact -- not evidence that a token ever appeared in
    training.  ``TokenIndex`` therefore uses canonical forms only.  The
    principled upgrade is corpus-observed ``text_mention`` forms (Stage 2), not
    aliases.

3.  **There is no clean stopword cutoff below the top.**  ``the`` occurs in
    2,024 entities and ``of`` in 830, then ``a`` at 209 -- so any threshold in
    [209, 830) removes exactly those two.  Below that the distribution is a
    smooth Zipf where ``john`` (180), ``david`` (138), ``county`` (137) are
    interleaved with ``and`` (163), ``in`` (147): any cutoff aggressive enough
    to catch the function words also deletes the collisions the hypothesis is
    about.  Hence ``DF_CUTOFF`` defaults to the gap, and ``idf_weighted=True``
    is offered as the threshold-free alternative.  Re-run
    ``token_document_frequency`` whenever the tokenizer changes -- BPE pieces
    have a different frequency profile than whole words.

Every number in (2) and (3) is recomputed and saved by section 12 of
``token_strain/strain_diagnostics.py``.  An earlier version of (2) quoted "5 to 251";
that pair does not reproduce under any denominator or cutoff and was replaced by the
figures above, while the direction it described holds throughout.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

# Removes exactly {the, of} on popqa-kas canonical forms: the observed gap
# between `of` (830 entities) and the next token `a` (209).  Not a curated
# stopword list -- see the module docstring for why one would be wrong.
DF_CUTOFF = 400

# Optimizer steps in one pass over the corpus, from batch_indices_epoch_*.npy
# (each file has exactly this many; 6 x 109,672 = 658,032 = the final 6E step).
STEPS_PER_EPOCH = 109_672

_WORD_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #
def word_tokens(text: str) -> set[str]:
    """Lower-cased alphanumeric words.  The cheap tokenisation, used for tests
    and for the diagnostic sweeps; ``bpe_tokens`` is what the model actually
    sees."""
    return set(_WORD_RE.findall(str(text).lower()))


def bpe_tokens(text: str, tokenizer, prefix_space: bool = True) -> set[str]:
    """Model-vocabulary tokens for ``text``.

    ``prefix_space`` mirrors ``data.answer_token_ids``: an answer is scored as
    ``" " + obj``, so its first piece carries the leading space and is a
    *different* token from the same word mid-sentence.  Keeping that distinction
    is the point -- competition at the first emitted token is where the decoding
    decision actually happens.
    """
    s = (" " + str(text)) if prefix_space else str(text)
    ids = tokenizer(s, add_special_tokens=False)["input_ids"]
    return {str(i) for i in ids}


def first_content_token(text: str, tokenizer, df: Counter | None = None,
                        prefix_space: bool = True) -> str | None:
    """The first token the model must emit for ``text``, skipping high-frequency
    pieces when ``df`` is supplied.

    This is the sharpest mechanistic variant: at the moment of generation the
    model commits to one token, and everything competing for *that* token is a
    live distractor.
    """
    s = (" " + str(text)) if prefix_space else str(text)
    ids = tokenizer(s, add_special_tokens=False)["input_ids"]
    for i in ids:
        t = str(i)
        if df is None or df.get(t, 0) <= DF_CUTOFF:
            return t
    return str(ids[0]) if ids else None


def token_document_frequency(qid_to_tokens: dict) -> Counter:
    """token -> number of distinct entities whose surface form contains it.

    This is the quantity the cutoff and the IDF weights are defined against, and
    the thing to re-inspect after any tokenizer change.
    """
    df = Counter()
    for toks in qid_to_tokens.values():
        for t in toks:
            df[t] += 1
    return df


# --------------------------------------------------------------------------- #
# Entity universe
# --------------------------------------------------------------------------- #
class EntityUniverse:
    """The set of entities that can compete, and the chunks each was seen in.

    Built from popqa-kas.  Two facts verified on the real table make this sound
    and are asserted at build time:

      * a QID carries the same chunk list regardless of whether it appears in
        the subject or the object role (0 inconsistencies over 19,384 entities);
      * ``shared_chunks == subject_chunks & answer_chunks`` exactly, for all
        14,267 facts.

    So one chunk set per QID is the whole story, and every windowed quantity
    below is set algebra over those.
    """

    def __init__(self, qid_to_form: dict, qid_to_chunks: dict):
        self.qid_to_form = dict(qid_to_form)
        self.qids = sorted(self.qid_to_form)
        self.index = {q: i for i, q in enumerate(self.qids)}
        self.chunks = {q: np.unique(np.asarray(c, dtype=np.int64))
                       for q, c in qid_to_chunks.items()}

    def __len__(self) -> int:
        return len(self.qids)

    @classmethod
    def from_popqa(cls, df, verify: bool = True) -> "EntityUniverse":
        """Build from a popqa-kas DataFrame (needs s_uri/o_uri, subj/obj and the
        three chunk columns)."""
        form, chunks = {}, {}

        def arr(v):
            return np.asarray([] if v is None else v, dtype=np.int64)

        for r in df.itertuples():
            for qid, txt, ch in ((r.s_uri, r.subj, r.subject_chunks),
                                 (r.o_uri, r.obj, r.answer_chunks)):
                if qid not in form:
                    form[qid] = str(txt)
                    chunks[qid] = arr(ch)
                elif verify and len(chunks[qid]) != len(arr(ch)):
                    raise ValueError(
                        f"{qid} has two different chunk lists "
                        f"({len(chunks[qid])} vs {len(arr(ch))}); the "
                        "one-chunk-set-per-entity assumption is violated"
                    )
        return cls(form, chunks)

    def tokens(self, tokenizer=None, prefix_space: bool = True) -> dict:
        """qid -> token set.  Word tokens when ``tokenizer`` is None, else BPE.

        Canonical surface form only, deliberately -- see module docstring (2).
        """
        if tokenizer is None:
            return {q: word_tokens(f) for q, f in self.qid_to_form.items()}
        return {q: bpe_tokens(f, tokenizer, prefix_space)
                for q, f in self.qid_to_form.items()}


# --------------------------------------------------------------------------- #
# Token -> entity index
# --------------------------------------------------------------------------- #
class TokenIndex:
    """Inverted index token -> entities, and the strain/ownership features.

    ``df_cutoff`` drops tokens carried by more than that many entities.  With
    ``idf_weighted``, a shared token contributes ``1/log(1+df)`` instead of 1,
    which avoids a threshold entirely; the two are reported side by side because
    neither is obviously right (see module docstring (3)).
    """

    def __init__(self, universe: EntityUniverse, qid_to_tokens: dict,
                 df_cutoff: int | None = DF_CUTOFF, idf_weighted: bool = False):
        self.u = universe
        self.df_full = token_document_frequency(qid_to_tokens)
        self.df_cutoff = df_cutoff
        self.idf_weighted = idf_weighted

        keep = (lambda t: True) if df_cutoff is None else (
            lambda t: self.df_full[t] <= df_cutoff)
        self.qid_to_tokens = {q: {t for t in ts if keep(t)}
                              for q, ts in qid_to_tokens.items()}

        self.token_to_qids: dict[str, set] = defaultdict(set)
        for q, ts in self.qid_to_tokens.items():
            for t in ts:
                self.token_to_qids[t].add(q)

        self._rivals: dict = {}

    # -- competitor sets --------------------------------------------------- #
    def rivals(self, qid) -> set:
        """Entities sharing >=1 (kept) token with ``qid``, excluding itself."""
        if qid not in self._rivals:
            out: set = set()
            for t in self.qid_to_tokens.get(qid, ()):
                out |= self.token_to_qids[t]
            out.discard(qid)
            self._rivals[qid] = out
        return self._rivals[qid]

    def strain(self, qid, exclude=()) -> float:
        """Competitor count for ``qid``.  ``exclude`` is the fact's *other*
        endpoint -- when scoring a subject, its own object is not a distractor.

        Unweighted (the default, and what every reported result uses) this is the
        **cardinality of the union** over the entity's tokens -- a DISTINCT count,
        neither a sum nor a mean.  An entity sharing "buffalo" with 2 rivals and
        "bills" with 3, one rival in both, scores 4: not 5 (sum) and not 2.5
        (mean).  Two consequences worth knowing:

          * it is monotone non-decreasing in token count -- another token can only
            add rivals -- which is the structural reason strain tracks name length;
          * it saturates, because rival sets of common tokens overlap heavily, so
            the union grows sub-linearly in the number of tokens.

        WARNING -- the ``idf_weighted`` branch is a **sum over tokens and does not
        deduplicate**: a rival reachable through two of the entity's tokens is
        counted twice, once per token weight.  So the two modes measure different
        things (distinct rival entities vs. weighted token-rival mass) and their
        magnitudes are not comparable.  Nothing reported uses the weighted mode;
        fix the double-counting before relying on it.
        """
        ex = set(exclude) | {qid}
        if not self.idf_weighted:
            return float(len(self.rivals(qid) - ex))
        total = 0.0
        for t in self.qid_to_tokens.get(qid, ()):
            n = len(self.token_to_qids[t] - ex)
            if n:
                total += n / np.log(1.0 + self.df_full[t])
        return total

    def rarest_token(self, qid) -> str | None:
        """The entity's most discriminative kept token.  Aggregating ownership
        over all tokens averages the informative token together with the junk;
        the rarest one is reported separately for that reason."""
        ts = self.qid_to_tokens.get(qid) or ()
        return min(ts, key=lambda t: self.df_full[t]) if ts else None

    def n_rivals_on(self, token) -> int:
        return max(len(self.token_to_qids.get(token, ())) - 1, 0)

    # -- Feature 1: ownership share ---------------------------------------- #
    def token_chunk_pool(self, token, chunk_filter=None) -> int:
        """|union of chunks over every entity carrying ``token``|.

        ``chunk_filter`` is a boolean callable over a chunk-id array, used to
        restrict the pool to one training window.
        """
        qs = self.token_to_qids.get(token, ())
        if not qs:
            return 0
        parts = []
        for q in qs:
            c = self.u.chunks.get(q)
            if c is None or c.size == 0:
                continue
            parts.append(c[chunk_filter(c)] if chunk_filter is not None else c)
        if not parts:
            return 0
        return int(np.unique(np.concatenate(parts)).size)

    def ownership(self, qid, n_shared: int, chunk_filter=None) -> dict:
        """Feature 1 for one endpoint of a fact.

        ``owned`` = this fact's evidence as a share of all evidence carrying the
        token.  Low means the fact's signal is a small slice of what the model
        saw for those tokens.
        """
        rare = self.rarest_token(qid)
        if rare is None:
            return {"own_rare": np.nan, "own_mean": np.nan,
                    "rivals_rare": 0, "rivals_max": 0}
        owned, rivals = [], []
        for t in self.qid_to_tokens[qid]:
            pool = self.token_chunk_pool(t, chunk_filter)
            owned.append(n_shared / pool if pool else np.nan)
            rivals.append(self.n_rivals_on(t))
        pool_rare = self.token_chunk_pool(rare, chunk_filter)
        return {
            "own_rare": (n_shared / pool_rare) if pool_rare else np.nan,
            "own_mean": float(np.nanmean(owned)) if owned else np.nan,
            "rivals_rare": self.n_rivals_on(rare),
            "rivals_max": max(rivals) if rivals else 0,
        }


# --------------------------------------------------------------------------- #
# Feature 2: alternative-answer co-occurrence
# --------------------------------------------------------------------------- #
# Laplace smoothing for the ratio.  Without it a fact with shared=1, distract=0
# scores a perfect 1.0, so the "cleanest" bucket fills up with facts that have
# almost no evidence at all -- which is what made the raw ratio run backwards.
RATIO_ALPHA = 1.0


def alt_answer_features(df, universe: EntityUniverse, chunk_filter=None,
                        alpha: float = RATIO_ALPHA) -> dict:
    """For each fact: how many chunks pair its subject with a *different* valid
    answer for the same relation.

    This is the data-side analogue of the paper's §6.3 counterfactual
    construction ("for 'The capital of France is Paris', 'Rome' also co-occurs
    with 'France'"): every such chunk is a chunk where the subject appears next
    to something the model could plausibly answer instead.

    Returns ``{fact_id: {"shared", "distract", "ratio", "evidence"}}``.
    ``ratio`` is smoothed; ``evidence = shared + distract`` is the exposure level
    it must always be reported beside -- unsmoothed and uncontrolled, ``distract``
    is mostly a proxy for "this subject is mentioned a lot".
    """
    out = {}
    for prop, g in df.groupby("prop", sort=False):
        objs = list(dict.fromkeys(g["o_uri"]))
        obj_set = set(objs)
        parts = [universe.chunks[o] for o in objs
                 if universe.chunks.get(o) is not None and universe.chunks[o].size]
        if parts:
            pool = np.concatenate(parts)
            uniq, cnt = np.unique(pool, return_counts=True)
        else:
            uniq = np.asarray([], dtype=np.int64)
            cnt = np.asarray([], dtype=np.int64)

        for r in g.itertuples():
            sc = universe.chunks.get(r.s_uri)
            oc = universe.chunks.get(r.o_uri)
            if sc is None:
                sc = np.asarray([], dtype=np.int64)
            if oc is None:
                oc = np.asarray([], dtype=np.int64)
            if chunk_filter is not None:
                sc = sc[chunk_filter(sc)]
                oc = oc[chunk_filter(oc)]

            if sc.size and uniq.size:
                i = np.clip(np.searchsorted(uniq, sc), 0, uniq.size - 1)
                n_alt = np.where(uniq[i] == sc, cnt[i], 0)
            else:
                n_alt = np.zeros(sc.size, dtype=np.int64)
            own = np.isin(sc, oc)                    # chunk holds the TRUE answer
            n_alt = n_alt - own.astype(np.int64)     # don't count it as a distractor
            # The subject is often itself a valid answer for this relation
            # (Buffalo is the object of "X is based in Buffalo" and the subject
            # of "Buffalo is in New York").  Every chunk in `sc` contains the
            # subject by construction, so without this it would be its own
            # distractor everywhere and `distract` would just count `sc`.
            if r.s_uri in obj_set:
                n_alt = n_alt - 1
            distract = int((n_alt > 0).sum())
            shared = int(own.sum())
            denom = shared + distract + 2 * alpha
            out[r.id] = {
                "shared": shared,
                "distract": distract,
                "evidence": shared + distract,
                # Undefined, not 1.0, when the subject was never seen with any
                # candidate answer -- there is no signal to be clean or dirty.
                "ratio": ((shared + alpha) / denom) if denom > 0 else float("nan"),
            }
    return out


# --------------------------------------------------------------------------- #
# Training-schedule windows
# --------------------------------------------------------------------------- #
class WindowMap:
    """Which training window each chunk was seen in, across all epochs.

    ``batch_indices_epoch_{e}.npy`` gives, for epoch ``e``, the chunk ids in each
    optimizer step.  Global step = ``(e-1) * steps_per_epoch + local_step``.  A
    chunk appears once per epoch at a *different* position each time -- OLMo-core
    reshuffles per epoch (``data_loader.py``: ``get_rng(seed + epoch + seq_len)``)
    -- so all six files are needed; epoch 1 alone covers only ~5% of the
    forgetting events.

    Stored as one bitmask per chunk (bit ``w`` set iff the chunk was seen in
    window ``w``), which is ~21 MB for the full corpus and gives O(1) lookup.
    """

    def __init__(self, chunk_window_mask: np.ndarray, boundaries: list[int]):
        self.mask = chunk_window_mask
        self.boundaries = list(boundaries)
        self.n_windows = len(self.boundaries) - 1

    @classmethod
    def from_batch_indices(cls, epoch_paths, boundaries, steps_per_epoch=None):
        """``epoch_paths`` in epoch order; ``boundaries`` are the checkpoint
        steps, so window ``w`` is ``(boundaries[w], boundaries[w+1]]``."""
        if len(boundaries) < 2:
            raise ValueError("need at least two checkpoint boundaries")
        n_win = len(boundaries) - 1
        if n_win > 63:
            raise ValueError(f"{n_win} windows exceeds the 63-bit mask")
        bounds = np.asarray(boundaries, dtype=np.int64)

        per_epoch = [np.load(p, allow_pickle=True) for p in epoch_paths]
        if steps_per_epoch is None:
            steps_per_epoch = len(per_epoch[0])

        # Rows come out of the object array with dtype=object when every batch
        # has the same length, so cast explicitly rather than trusting asarray.
        def _ids(x):
            return np.asarray(x).ravel().astype(np.int64)

        max_chunk = 0
        for bi in per_epoch:
            for ids in bi:
                a = _ids(ids)
                if a.size:
                    max_chunk = max(max_chunk, int(a.max()))
        mask = np.zeros(max_chunk + 1, dtype=np.int64)

        for e, bi in enumerate(per_epoch):
            for local_step, ids in enumerate(bi):
                a = _ids(ids)
                if not a.size:
                    continue
                gstep = e * steps_per_epoch + local_step
                # window w covers (boundaries[w], boundaries[w+1]]
                w = int(np.searchsorted(bounds, gstep, side="left")) - 1
                if gstep == bounds[0]:
                    continue                      # step 0 precedes every window
                if 0 <= w < n_win:
                    mask[a] |= np.int64(1) << np.int64(w)
        return cls(mask, boundaries)

    def in_window(self, chunk_ids, w: int) -> np.ndarray:
        """Boolean mask over ``chunk_ids`` -- was each seen during window ``w``?"""
        a = np.asarray(chunk_ids, dtype=np.int64)
        if a.size == 0:
            return np.zeros(0, dtype=bool)
        safe = np.clip(a, 0, self.mask.size - 1)
        got = (self.mask[safe] >> np.int64(w)) & np.int64(1)
        return (got == 1) & (a < self.mask.size) & (a >= 0)

    def filter_for(self, w: int):
        """A ``chunk_filter`` callable for the feature functions above."""
        return lambda chunk_ids: self.in_window(chunk_ids, w)

    def active_entities(self, universe: EntityUniverse, w: int) -> set:
        """Entities with >=1 chunk seen during window ``w``."""
        return {q for q, c in universe.chunks.items()
                if c.size and bool(self.in_window(c, w).any())}


def windowed_strain(index: TokenIndex, qid, active: set, exclude=()) -> float:
    """Strain restricted to entities that actually appeared in the window.

    This is the form the hypothesis is really about -- "how many competing
    entities has the model seen *since the last checkpoint*" -- and the only
    form with a time dimension.  The static version cannot explain *when* a fact
    is forgotten, only which facts are.
    """
    ex = set(exclude) | {qid}
    live = (index.rivals(qid) & active) - ex
    if not index.idf_weighted:
        return float(len(live))
    total = 0.0
    for t in index.qid_to_tokens.get(qid, ()):
        n = len((index.token_to_qids[t] & active) - ex)
        if n:
            total += n / np.log(1.0 + index.df_full[t])
    return total
