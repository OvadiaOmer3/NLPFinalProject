"""Held-out in-context demonstrations for the PopQA cloze eval.

Why this exists instead of ``task.fewshot_context``
---------------------------------------------------
``PopQACloze`` sets no ``fewshot_source`` and reports no training/validation
split, so olmes' ``fewshot_examples`` lands on::

    return rnd.sample(self._fewshot_docs, k * 2)   # base_task.py:597
    #      ^ self.test_docs() -- the set being scored

Demonstrations are therefore drawn from the eval set, with only the doc under
evaluation filtered out.  Other eval facts still appear in context *with their
gold answers*, so ``correct`` partly measures in-context copying rather than
parametric recall -- fatal for an acquisition-timeline study.

Why not the fork's curated set either
-------------------------------------
``FEWSHOT_SOURCES["popqa"]`` (vendored in ``fewshot_source.py``) is 16 hand-pasted
facts, one per ``prop_id``, with no generating script.  Measured against
popqa-kas it has three concrete problems:

  * ``prop 257``: "The father of Qasim ibn Muhammad ibn Abi Bakr is" ->
    "Muhammad ibn Abi Bakr" -- the answer is a substring of the question.  This is
    exactly what ``is_excluded`` drops from the *eval* set, shipped as a *demo*:
    it teaches "copy a span out of the question".
  * ``prop 182`` -> "United States of America" (284 eval facts share that answer)
    and ``prop 560`` -> "association football" (327) prime very common answers.
  * all 16 are themselves popqa-kas rows, and 20 eval facts share their subjects.

And ``num_shots=15`` over 16 examples silently drops the 16th, so ``prop 422``
("The capital of X is") is never demonstrated while still being evaluated.

Three demonstration modes
-------------------------
``FewShot`` renders the same pool three ways (``--fewshot-mode``), so the effect of
*what* is demonstrated can be measured rather than assumed:

  * ``drop-same-prop`` (default) -- a fact of relation ``p`` sees the pool minus
    relation ``p``: 15 demos of 15 *other* relations.  The answer type is never
    handed over.
  * ``fixed-block`` -- one constant block for every fact, its own relation
    included (upstream's behaviour).
  * ``only-same-prop`` -- the opposite extreme: every demo shares the query's
    relation, and no demo's answer would be accepted as the query's answer.  This
    isolates "same relation" from "same answer": the model gets the answer *type*
    for free but can never copy the answer itself.  It needs a much larger pool
    (``num_shots``+ demos per relation, not one), selected by ``select_same_prop_demos``.

Selection rule
--------------
Deterministic, no RNG.  Candidates must satisfy, and are then ranked by:

  1. ``not is_excluded`` -- the answer (any alias, len > 3) is absent from the
     query, so no demo rewards copying from the question.
  2. the subject owns exactly ONE fact, and its *name* is unique in the data --
     holding it out costs a single eval row and removes that subject from the eval
     set entirely, so no evaluated fact ever sees its own subject in context.
  3. ``len(subj) >= MIN_SUBJ_LEN`` -- short names ("magi", "Party", "Stuff") are
     substrings of many unrelated eval subjects, which would put an eval fact's
     subject string in the context attached to somebody else's answer.  At 12
     characters the measured collision count on popqa-kas is 0; at 0 it is 15.
  4. neither of ``subj``/``obj`` contains the other (case-insensitive) -- rejects
     near-tautologies like "Vaxholm is the capital of" -> "Vaxholm Municipality".
  5. the answer is unique *within the eval set* (``obj`` shared by <=
     ``OBJ_FREQ_MAX`` = 1 fact, i.e. only the demo itself), so no eval fact can be
     answered by copying a demo's answer.
  6. both entities are real enough to be unambiguous (``s_pop``/``o_pop`` sitelink
     counts >= ``MIN_POP``), then ranked by ``o_pop`` descending.

Rank-by-``o_pop`` matters: filters 1-5 alone leave only the rarest facts, which
are disproportionately junk ("The sport run batted in plays is" -> softball,
"The color of ruby is" -> pink).  Preferring a notable *answer* entity picks
coherent, correct-looking demonstrations instead.  Note ``obj_n`` (frequency in
the eval set) and ``o_pop`` (Wikipedia sitelinks) are independent, so 5 and 6
don't fight: the answer is globally notable but locally unique.

On popqa-kas this yields 16 demonstrations (one per relation) costing 16 eval
facts, with zero subject-string collisions and zero shared answers.

The pool is chosen from the FULL dataset before any ``--limit`` /
``--min-shared-chunks`` filtering, and its subjects are dropped from the eval set
UNCONDITIONALLY -- including at 0-shot -- so the eval set is byte-identical across
shot counts and 0-shot vs 15-shot is a clean comparison.

That holds *within* a mode.  ``only-same-prop`` holds out a bigger pool (~300 facts
vs 16), so its eval set is a subset of the other modes' -- compare modes on the
``id`` intersection, not on the raw means.
"""
from __future__ import annotations

import config
from src import lment_popqa

# Selection thresholds.  Tuned on popqa-kas so the resulting pool has ZERO
# measured overlap with the eval set (see the module docstring); they still leave
# enough candidates to cover all 16 relations.
OBJ_FREQ_MAX = 1        # max eval facts sharing a demo's answer
MIN_POP = 500           # min sitelink count for both demo entities
MIN_SUBJ_LEN = 12       # min demo subject length, in characters

# Thresholds for the same-relation pool (mode "only-same-prop"), which needs
# ~num_shots demos PER RELATION instead of one, i.e. ~20x the candidates.
#
# ``OBJ_FREQ_MAX`` is dropped entirely here, not merely relaxed: it cannot be
# satisfied.  A relation with a small answer range has no rare answers at all --
# under the strict rule popqa-kas leaves 1 candidate for prop 472 ("The color of X
# is"), 2 for prop 182 and 3 for prop 560, against the 16+ this mode needs.  Its job
# (no eval fact can be answered by copying a demo's answer) is done per-fact instead,
# and exactly: a demo is dropped for a query iff its answer would be *scored* as that
# query's answer (``answers_clash``).  That is the stronger guarantee -- the global
# rule only bounded how many facts *shared* an answer with the pool.
#
# ``MIN_SUBJ_LEN`` drops 12 -> 10 and is backed up by ``_subject_collides``, which
# checks the actual property the length heuristic was standing in for (the demo's
# subject string does not occur inside any other fact's subject).  ``MIN_POP`` drops
# 500 -> 100 for headroom; with these, every one of the 16 relations has >= 23
# candidates, so a 20-per-relation pool is reachable everywhere the answer range
# allows it -- what actually binds for a narrow relation is the per-answer cap below.
SAME_PROP_MIN_POP = 100
SAME_PROP_MIN_SUBJ_LEN = 10
# Demos per relation are num_shots + this, so a query whose answer collides with a
# pool answer still has num_shots demos left after the per-fact filter.
SAME_PROP_MARGIN = 5
# Max demos per relation carrying the SAME answer.  Relations differ enormously in
# answer range -- among popqa-kas candidates prop 484 ("The author of X is") has 468
# distinct answers while prop 472 ("The color of X is") has 4, 21 of whose 24
# candidates are "red".  Uncapped, a 20-demo colour bench is 17x "red": a query
# whose answer is black then gets a legal block (no answer clash) that is also a
# 15-fold vote for "red", which measures in-context bias, not the answer type.  At 2
# the thin relations simply run out of demos -- the shot count is recorded per fact
# (``num_shots`` in the output) and reported at build time, so the shortfall is
# visible instead of silently reshaping the prompt.
SAME_PROP_MAX_PER_ANSWER = 2


def select_demos(df, per_prop: int = 1, obj_freq_max: int = OBJ_FREQ_MAX,
                 min_pop: int = MIN_POP, min_subj_len: int = MIN_SUBJ_LEN):
    """Pick ``per_prop`` demonstrations for each ``prop_id`` -> a DataFrame.

    Deterministic given ``df``'s contents (ties broken by ``id``), so the pool is
    stable across checkpoints and across runs.  See the module docstring for the
    rule and why each clause is there.
    """
    subj_id_n = df["subj_id"].value_counts()
    subj_str_n = df["subj"].value_counts()
    obj_n = df["obj"].value_counts()

    c = df[~df.apply(lment_popqa.is_excluded, axis=1)].copy()
    contains = c.apply(
        lambda r: (str(r["subj"]).lower() in str(r["obj"]).lower()
                   or str(r["obj"]).lower() in str(r["subj"]).lower()), axis=1)
    c = c[(c["subj_id"].map(subj_id_n) == 1)        # one fact per demo subject
          & (c["subj"].map(subj_str_n) == 1)        # ...and the name is unique too
          & (c["obj"].map(obj_n) <= obj_freq_max)
          & (c["subj"].str.len() >= min_subj_len)
          & (~contains)
          & (c["s_pop"] >= min_pop)
          & (c["o_pop"] >= min_pop)]

    used_answers, picked = set(), []
    for prop in sorted(c["prop_id"].unique()):
        g = (c[(c["prop_id"] == prop) & (~c["obj"].isin(used_answers))]
             .sort_values(["o_pop", "id"], ascending=[False, True]))
        for _, r in g.head(per_prop).iterrows():
            used_answers.add(r["obj"])       # never demo the same answer twice
            picked.append(r)
    if not picked:
        raise ValueError(
            "no demonstration candidates survived selection; loosen "
            "fewshot.OBJ_FREQ_MAX / MIN_POP / MIN_SUBJ_LEN"
        )
    import pandas as pd
    return pd.DataFrame(picked).reset_index(drop=True)


def _subject_collides(subj: str, subjects) -> bool:
    """True if ``subj`` occurs inside some *other* subject string in ``subjects``.

    ``subjects`` is a set of lower-cased subject strings (the whole dataset's).  This
    is the property ``MIN_SUBJ_LEN`` approximates: a demo subject that is a substring
    of an eval subject puts that eval fact's subject into somebody else's context,
    attached to the wrong answer ("Party" inside "Communist Party of China").  Checked
    directly, the length floor can be lower without reintroducing the problem.
    """
    s = str(subj).lower()
    return any(s in other for other in subjects if other != s)


def select_same_prop_demos(df, per_prop: int, min_pop: int = SAME_PROP_MIN_POP,
                           min_subj_len: int = SAME_PROP_MIN_SUBJ_LEN,
                           max_per_answer: int = SAME_PROP_MAX_PER_ANSWER,
                           verbose: bool = True):
    """Pick ``per_prop`` demonstrations for EVERY ``prop_id`` -> a DataFrame.

    The pool for mode ``only-same-prop``, where a query is shown demos of its own
    relation only, so each relation needs its own bench of ~``num_shots`` facts.

    Same hygiene filters as ``select_demos`` (answer not in the question, demo subject
    owns one fact and a unique name, no subject/answer containment, both entities
    popular enough) minus the global answer-uniqueness rule, which is unsatisfiable
    here -- see the ``SAME_PROP_*`` comments above.

    Within a relation, candidates are ordered *answer-diversity first*: all
    distinct answers (each the most notable fact carrying it) before any answer's
    second fact, and at most ``max_per_answer`` facts per answer.  A relation with a
    small answer range therefore returns fewer than ``per_prop`` demos rather than
    the same answer over and over -- see ``SAME_PROP_MAX_PER_ANSWER``.
    Deterministic: ties break on ``id``, no RNG.
    """
    import pandas as pd

    subj_id_n = df["subj_id"].value_counts()
    subj_str_n = df["subj"].value_counts()
    all_subjects = set(df["subj"].astype(str).str.lower())

    c = df[~df.apply(lment_popqa.is_excluded, axis=1)].copy()
    contains = c.apply(
        lambda r: (str(r["subj"]).lower() in str(r["obj"]).lower()
                   or str(r["obj"]).lower() in str(r["subj"]).lower()), axis=1)
    c = c[(c["subj_id"].map(subj_id_n) == 1)        # one fact per demo subject
          & (c["subj"].map(subj_str_n) == 1)        # ...and the name is unique too
          & (c["subj"].str.len() >= min_subj_len)
          & (~contains)
          & (c["s_pop"] >= min_pop)
          & (c["o_pop"] >= min_pop)]
    # Rank by notability, then re-rank answer-diversity first: `_dup` is how many
    # better-ranked candidates of this relation already carry the same answer.
    c = c.sort_values(["o_pop", "id"], ascending=[False, True])
    c = c.assign(_dup=c.groupby(["prop_id", "obj"]).cumcount())
    c = c[c["_dup"] < max(max_per_answer, 1)]        # the per-answer cap

    picked, short = [], {}
    for prop in sorted(c["prop_id"].unique()):
        g = (c[c["prop_id"] == prop]
             .sort_values(["_dup", "o_pop", "id"], ascending=[True, False, True]))
        n = 0
        # The collision check is O(dataset) per candidate, so it runs lazily on the
        # ranked list and stops at per_prop instead of filtering all ~5k candidates.
        for _, r in g.iterrows():
            if _subject_collides(r["subj"], all_subjects):
                continue
            picked.append(r)
            n += 1
            if n >= per_prop:
                break
        if n < per_prop:
            short[int(prop)] = n
    if not picked:
        raise ValueError(
            "no same-relation demonstration candidates survived selection; loosen "
            "fewshot.SAME_PROP_MIN_POP / SAME_PROP_MIN_SUBJ_LEN"
        )
    if short and verbose:
        print(f"[fewshot] {len(short)} relations have fewer than {per_prop} "
              f"demonstrations available (answer range too narrow at "
              f"{max_per_answer} demos per answer) -- prop_id: available = {short}")
    return (pd.DataFrame(picked).drop(columns="_dup").reset_index(drop=True))


def answer_strings(row) -> list[str]:
    """Everything that counts as ``row``'s answer, lower-cased.

    olmes' scoring labels (``possible_answers`` with ``len > 3``) plus the raw
    ``obj``: the length filter is upstream's, and it drops exactly the short answers
    this project's smallest-range relation is made of ("red" for prop 472), which we
    still must not demonstrate to a query whose answer is "red".
    """
    labels = [str(l).lower() for l in lment_popqa.answer_labels(row)]
    labels.append(str(row["obj"]).lower())
    return sorted({l for l in labels if l})


def _labels_clash(a: list[str], b: list[str]) -> bool:
    """Substring overlap either way between two ``answer_strings`` lists."""
    return any(x in y or y in x for x in a for y in b)


def answers_clash(target, demo) -> bool:
    """True if ``demo``'s answer would be accepted as ``target``'s answer.

    Mirrors ``lment_popqa.is_correct``'s substring rule in both directions (so
    "United States" vs "United States of America" clashes whichever side it is on)
    and additionally treats a shared ``obj_id`` as a clash, which catches aliases
    that share no substring at all.  Demonstrations that clash are what
    ``only-same-prop`` filters out per query.
    """
    try:
        if target["obj_id"] == demo["obj_id"]:
            return True
    except KeyError:
        pass
    return _labels_clash(answer_strings(target), answer_strings(demo))


def _fix_surrogates(v):
    """Recombine UTF-16 surrogate pairs into real characters.

    The vendored upstream demos carry ``"\\ud83c\\uddf5\\ud83c\\uddea"`` (the 🇵🇪
    flag) inside ``possible_answers`` -- written as a Python literal, so it parses
    to *lone surrogates* rather than the emoji.  pandas/pyarrow then refuse to
    encode the column ("surrogates not allowed") and DataFrame construction dies.
    """
    if not isinstance(v, str):
        return v
    try:
        return v.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeError:
        return v.encode("utf-8", "replace").decode("utf-8")


def upstream_demos():
    """The fork's ``FEWSHOT_SOURCES["popqa"]`` as a DataFrame -- for reproducing
    LMEnt's numbers as-is, leakage included.  See ``--fewshot-source upstream``."""
    import pandas as pd
    from src.fewshot_source import FEWSHOT_POPQA
    return pd.DataFrame([{k: _fix_surrogates(v) for k, v in d.items()}
                         for d in FEWSHOT_POPQA])


MODES = ("drop-same-prop", "fixed-block", "only-same-prop")


class FewShot:
    """A demonstration pool, rendered into the prefix(es) prompts are built from.

    Three modes (see the module docstring for the comparison they enable):

    ``drop-same-prop`` (default)
        A fact of relation ``p`` is shown the pool MINUS its own relation: with a
        one-per-relation pool that is exactly ``len(pool) - 1`` = 15 demonstrations,
        and every pool fact is used somewhere in the eval, so nothing is wasted the
        way upstream's ``[:15]`` of 16 is.  The point is not the count -- it's that
        the relation under test is never demonstrated.  A same-relation demo supplies
        the answer *type* ("The color of X is <a colour>"), which is in-context
        scaffolding for precisely the thing an acquisition study wants to read out of
        the weights.  At most ``n_relations`` distinct prefixes (16 here).

    ``fixed-block``
        One constant block for every fact, its own relation included -- upstream's
        behaviour.  Exactly one prefix.

    ``only-same-prop``
        Every demonstration shares the query's relation, and every demonstration
        whose answer would be *scored* as the query's answer is dropped
        (``answers_clash``).  So the answer type is handed over deliberately while
        copying stays impossible -- the contrast with ``drop-same-prop`` measures what
        that type information is worth.  Needs a ``select_same_prop_demos`` pool;
        prefixes vary per query (by which demos its answer knocked out), so they are
        rendered lazily and cached by demo-id tuple.
    """

    def __init__(self, demos, num_shots: int, mode: str = "drop-same-prop"):
        if mode not in MODES:
            raise ValueError(f"unknown few-shot mode {mode!r}; expected one of {MODES}")
        self.num_shots = num_shots
        self.demos = demos                      # the whole pool
        self.mode = mode
        # Subjects of the WHOLE pool, whatever the shot count: the eval set must
        # not shift with --num-shots, or shot settings stop being comparable.
        self.subject_ids = set(demos["subj_id"])
        self._render_cache: dict = {}

        # only-same-prop walks its relation's bench per query, so keep it as plain
        # row lists (pool order = answer-diversity order) and pre-extract the answer
        # strings the clash test needs.
        self._pool_by_prop: dict = {}
        self._answers_by_demo: dict = {}
        if mode == "only-same-prop":
            for p, g in demos.groupby("prop_id", sort=True):
                self._pool_by_prop[int(p)] = [r for _, r in g.iterrows()]
            self._answers_by_demo = {r["id"]: answer_strings(r)
                                     for _, r in demos.iterrows()}

    @classmethod
    def build(cls, df, num_shots: int | None = None, source: str = "ours",
              per_prop: int | None = None, mode: str = "drop-same-prop",
              verbose: bool = True) -> "FewShot":
        if mode not in MODES:
            raise ValueError(f"unknown few-shot mode {mode!r}; expected one of {MODES}")
        num_shots = config.NUM_SHOTS if num_shots is None else num_shots

        if mode == "only-same-prop":
            if source != "ours":
                raise ValueError(
                    "mode only-same-prop needs --fewshot-source ours: the upstream "
                    "pool holds a single fact per relation, so there is nothing "
                    "same-relation left to demonstrate."
                )
            # One bench per relation, with margin for the demos a query's own answer
            # knocks out.
            per_prop = num_shots + SAME_PROP_MARGIN if per_prop is None else per_prop
            demos = select_same_prop_demos(df, per_prop=per_prop, verbose=verbose)
            # A query only ever sees its own relation's bench, so the whole pool is
            # not the capacity -- the biggest bench is.  Deliberately not the
            # *thinnest*: a relation with a narrow answer range (prop 472, four
            # distinct colours) can never fill 15 slots, and hard-failing on it would
            # make the mode unusable at any realistic shot count.  Those facts get
            # what their relation has, recorded per fact and reported below.
            capacity = int(demos["prop_id"].value_counts().max())
        else:
            per_prop = 1 if per_prop is None else per_prop
            demos = upstream_demos() if source == "upstream" else select_demos(
                df, per_prop=per_prop)
            # Available slots depend on the mode: dropping the query's own relation
            # costs `per_prop` demonstrations out of the pool.
            capacity = len(demos) - (per_prop if mode == "drop-same-prop" else 0)

        if num_shots > capacity:
            why = {
                "only-same-prop": ", the biggest relation's bench",
                "drop-same-prop": f", minus {per_prop} for the query's own relation",
            }.get(mode, "")
            raise ValueError(
                f"--num-shots {num_shots} exceeds {capacity} available "
                f"demonstrations (pool={len(demos)}, source={source}, mode={mode}"
                f"{why}). Raise --fewshot-per-prop, lower --num-shots, or pick "
                "another --fewshot-mode."
            )
        fs = cls(demos, num_shots, mode=mode)
        if verbose:
            n_prop = demos["prop_id"].nunique()
            print(f"[fewshot] source={source} mode={mode} num_shots={num_shots} "
                  f"pool={len(demos)} facts over {n_prop} relations")
            if mode == "only-same-prop" and num_shots > 0:
                # Cheap (no rendering): report up front whether any query ends up with
                # fewer than num_shots demos, and which relations are responsible.
                # Measured on the eval set, i.e. after the pool's own facts leave.
                ev = df[~df["subj_id"].isin(fs.subject_ids)]
                shots = ev.apply(fs.shots_for, axis=1)
                short = shots < num_shots
                print(f"[fewshot] same-relation demos per fact: min={int(shots.min())} "
                      f"mean={shots.mean():.2f}; {int(short.sum())} of {len(ev)} facts "
                      f"get < {num_shots}")
                if short.any():
                    by_prop = (shots[short].groupby(ev.loc[short, "prop_id"])
                               .agg(["size", "min", "max"]))
                    for prop, (n, lo, hi) in by_prop.iterrows():
                        print(f"[fewshot]   prop {int(prop)}: {int(n)} facts at "
                              f"{int(lo)}-{int(hi)} demos (its bench has "
                              f"{len(fs._pool_by_prop.get(int(prop), []))} facts; the "
                              "rest share the query's answer)")
            elif num_shots < capacity:
                print(f"[fewshot] WARNING: --num-shots {num_shots} < {capacity} "
                      "available, so the highest prop_ids are never demonstrated "
                      "(demonstrations are truncated in prop_id order)")
        return fs

    # ------------------------------------------------------------------ #
    # Demonstration selection per query
    # ------------------------------------------------------------------ #
    def _head(self, sel) -> list:
        return [r for _, r in sel.head(max(self.num_shots, 0)).iterrows()]

    def _same_prop_demos(self, row) -> list:
        """This relation's bench minus every demo that gives away ``row``'s answer."""
        pool = self._pool_by_prop.get(int(row["prop_id"]), [])
        target = answer_strings(row)
        obj_id = row["obj_id"] if "obj_id" in row else None
        kept = []
        for d in pool:
            if obj_id is not None and d["obj_id"] == obj_id:
                continue
            if _labels_clash(target, self._answers_by_demo[d["id"]]):
                continue
            kept.append(d)
            if len(kept) >= self.num_shots:
                break
        return kept

    def demos_for(self, row) -> list:
        """The demonstrations ``row`` is actually shown, in order (for inspection;
        ``prefix_for`` renders exactly this list)."""
        if self.num_shots <= 0:
            return []
        if self.mode == "fixed-block":
            return self._head(self.demos)
        if self.mode == "drop-same-prop":
            return self._head(self.demos[self.demos["prop_id"] != row["prop_id"]])
        return self._same_prop_demos(row)

    def shots_for(self, row) -> int:
        """How many demonstrations ``row`` actually gets.  Equals ``num_shots``
        except in ``only-same-prop``, where a query's answer can knock out enough of
        its relation's bench to leave fewer."""
        return len(self.demos_for(row))

    def prefix_for_demos(self, demos) -> str:
        """Render a demonstration block, cached by the tuple of demo ids.

        Split out of ``prefix_for`` for callers that already hold the demo list
        (``src.prompt_set.PromptSet.build``), so ``demos_for`` -- which under
        ``only-same-prop`` walks the relation's bench per query -- is not paid for
        twice.
        """
        key = tuple(d["id"] for d in demos)
        prefix = self._render_cache.get(key)
        if prefix is None:
            prefix = lment_popqa.render_fewshot_block(demos)
            self._render_cache[key] = prefix
        return prefix

    def prefix_for(self, row) -> str:
        """Demonstration block for ``row`` ("" at 0-shot).

        Cached by the tuple of demo ids, so the number of renders is the number of
        *distinct* demo sets (1 for fixed-block, one per relation for
        drop-same-prop, one per relation x knocked-out set for only-same-prop) --
        never one per fact.
        """
        return self.prefix_for_demos(self.demos_for(row))

    def prompt(self, row) -> str:
        """Full recall prompt for ``row``: demo block + this fact's cloze."""
        return lment_popqa.make_prompt(row, prefix=self.prefix_for(row))

    def drop_subjects(self, df, verbose: bool = True):
        """Remove eval facts whose subject is used by any demonstration."""
        mask = df["subj_id"].isin(self.subject_ids)
        if verbose:
            print(f"[fewshot] dropped {int(mask.sum())} eval facts on "
                  f"{len(self.subject_ids)} demonstration subjects "
                  f"({len(df) - int(mask.sum())} remain)")
        return df[~mask].reset_index(drop=True)
