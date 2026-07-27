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


class FewShot:
    """A demonstration pool, rendered into the prefix(es) prompts are built from.

    With ``drop_same_prop`` (the default) a fact of relation ``p`` is shown the
    pool MINUS its own relation: with a one-per-relation pool that is exactly
    ``len(pool) - 1`` = 15 demonstrations, and every pool fact is used somewhere in
    the eval, so nothing is wasted the way upstream's ``[:15]`` of 16 is.

    The point is not the count -- it's that the relation under test is never
    demonstrated.  A same-relation demo supplies the answer *type* ("The color of
    X is <a colour>"), which is in-context scaffolding for precisely the thing an
    acquisition study wants to read out of the weights.  Every fact is still
    treated symmetrically: 15 demonstrations, none sharing its relation.

    So there are at most ``n_relations`` distinct prefixes (16 here), each rendered
    once -- not one per fact.  Set ``drop_same_prop=False`` for a single constant
    block, which is what upstream does.
    """

    def __init__(self, demos, num_shots: int, drop_same_prop: bool = True):
        self.num_shots = num_shots
        self.demos = demos                      # the whole pool
        self.drop_same_prop = drop_same_prop
        # Subjects of the WHOLE pool, whatever the shot count: the eval set must
        # not shift with --num-shots, or shot settings stop being comparable.
        self.subject_ids = set(demos["subj_id"])

        def render(sel):
            return lment_popqa.render_fewshot_block(
                [r for _, r in sel.head(max(num_shots, 0)).iterrows()])

        self._default_prefix = render(demos)
        self._prefix_by_prop: dict = {}
        if drop_same_prop and num_shots > 0:
            for p in sorted(set(demos["prop_id"])):
                self._prefix_by_prop[p] = render(demos[demos["prop_id"] != p])

    @classmethod
    def build(cls, df, num_shots: int | None = None, source: str = "ours",
              per_prop: int = 1, drop_same_prop: bool = True,
              verbose: bool = True) -> "FewShot":
        demos = upstream_demos() if source == "upstream" else select_demos(
            df, per_prop=per_prop)
        # Available slots depend on the mode: dropping the query's own relation
        # costs `per_prop` demonstrations out of the pool.
        capacity = len(demos) - (per_prop if drop_same_prop else 0)
        num_shots = config.NUM_SHOTS if num_shots is None else num_shots
        if num_shots > capacity:
            raise ValueError(
                f"--num-shots {num_shots} exceeds {capacity} available "
                f"demonstrations (pool={len(demos)}, source={source}"
                + (f", minus {per_prop} for the query's own relation)"
                   if drop_same_prop else ")")
                + ". Raise --fewshot-per-prop, lower --num-shots, or pass "
                  "--fewshot-keep-same-prop."
            )
        fs = cls(demos, num_shots, drop_same_prop=drop_same_prop)
        if verbose:
            mode = "drop-same-prop" if drop_same_prop else "fixed-block"
            print(f"[fewshot] source={source} mode={mode} num_shots={num_shots} "
                  f"pool={len(demos)} facts over "
                  f"{demos['prop_id'].nunique()} relations")
            if num_shots < capacity:
                print(f"[fewshot] WARNING: --num-shots {num_shots} < {capacity} "
                      "available, so the highest prop_ids are never demonstrated "
                      "(demonstrations are truncated in prop_id order)")
        return fs

    def prefix_for(self, row) -> str:
        """Demonstration block for ``row``'s relation ("" at 0-shot)."""
        if not self.drop_same_prop:
            return self._default_prefix
        return self._prefix_by_prop.get(row["prop_id"], self._default_prefix)

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
