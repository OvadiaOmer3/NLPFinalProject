"""Single home for the olmes (LMEnt eval-harness) coupling.

Everything about *how the paper builds prompts and scores answers* lives here so
the rest of the project imports one clean API and stays readable.

Prompt construction and scoring are single-sourced from LMEnt's olmes fork
(``dhgottesman/olmes``, task ``PopQACloze`` in
``oe_eval/tasks/oe_eval_tasks/popqa_cloze.py``):

  * cloze prompts (models are not instruction-tuned) keyed by ``prop_id``;
  * few-shot context (default 15-shot): this module only *renders* the block (see
    ``render_fewshot_block``); which facts to demonstrate is ``src.fewshot``'s job,
    and deliberately is NOT upstream's sampling from the eval set;
  * exact-match scoring: labels = ``possible_answers`` with ``len > 3``, the
    example is *excluded* if a label is already in the query, otherwise correct
    iff any label substring-matches the generation (case-insensitive variants);
  * generation kwargs: ``max_gen_toks=15``, greedy, ``stop=["\\n\\n"]``.

If the olmes package is importable (submodule checked out + ``pip install -e
olmes``) we defer to the real ``PopQACloze`` task -- that path is authoritative.
Otherwise we use the faithful fallback below, transcribed from the fork, so the
prompt/scoring-parity checks in the plan run without installing olmes.

VERIFIED against commit 1f0568fb047b71651193dc832625e6f5697ba777 (the SHA LMEnt
pins for the olmes submodule): ``PopQACloze.TASK_CONFIG_DEFAULTS`` (num_shots=15,
split=test, the generation kwargs below), ``doc_to_text`` -> ``doc["query"]``,
``doc_to_target`` -> ``" " + gold``, and ``Task.fewshot_context`` /
``Task.fewshot_examples``.  One divergence is deliberate and documented in
``config`` / ``src.fewshot``: demonstrations come from a held-out pool instead of
``rnd.sample(self.test_docs(), k * 2)``, which would leak gold answers from the
set being scored.
"""
from __future__ import annotations

import functools
import json

import config

# --------------------------------------------------------------------------- #
# Transcribed from dhgottesman/olmes  popqa_cloze.py  (authoritative fallback)
# --------------------------------------------------------------------------- #
# POPQA_TEMPLATES is keyed by PopQA property id (`prop_id`), NOT the relation
# string.  Each template has a single `{}` slot for the subject.
POPQA_TEMPLATES: dict[int, str] = {
    22:  "The occupation of {} is",
    218: "The city {} was born in is",
    91:  "The genre of {} is",
    257: "The father of {} is",
    182: "The country {} is in is",
    164: "The producer of {} is",
    526: "The director of {} is",
    # 97 is "capital of" (subj=city, obj=the polity it is capital of) -- the
    # INVERSE of 422 ("capital", subj=polity, obj=city).  Both were once mapped
    # to "The capital of {} is", which asked 97's 363 facts the wrong question.
    97:  "{} is the capital of",
    533: "The screenwriter for {} is",
    639: "The composer of {} is",
    472: "The color of {} is",
    106: "The religion of {} is",
    560: "The sport {} plays is",
    484: "The author of {} is",
    292: "The mother of {} is",
    422: "The capital of {} is",
}

# olmes PopQACloze.TASK_CONFIG_DEFAULTS["generation_kwargs"]
GENERATION_KWARGS = {
    "max_gen_toks": 15,
    "temperature": 0.0,
    "do_sample": False,
    "stop_sequences": ["\n\n"],
}


# --------------------------------------------------------------------------- #
# Optional real-olmes task (authoritative when available)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _olmes_task():
    """Return an instantiated PopQACloze task, or None if olmes isn't installed."""
    try:
        from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY  # type: ignore
    except Exception:
        return None
    # VERIFY registry key at checkout (expected "popqa_cloze").
    for key in ("popqa_cloze", "popqa"):
        cls = TASK_REGISTRY.get(key)
        if cls is not None:
            task = cls(task_name=key, task_config={"num_shots": config.NUM_SHOTS})
            try:
                task.download()
            except Exception:
                pass
            return task
    return None


# --------------------------------------------------------------------------- #
# Row -> OLMES doc adapter
# --------------------------------------------------------------------------- #
def _alias(row) -> list[str]:
    """possible_answers as a list. akariasai/PopQA stores it as a JSON string;
    popqa-kas may already expose a list/array -- handle both."""
    pa = row["possible_answers"]
    if isinstance(pa, str):
        try:
            return list(json.loads(pa))
        except Exception:
            return [pa]
    return list(pa)


def process_doc(row) -> dict:
    """popqa-kas row -> the OLMES doc dict PopQACloze._process_doc produces."""
    prop_id = int(row["prop_id"])
    template = POPQA_TEMPLATES.get(prop_id, "{}")
    return {
        "id": row["id"],
        "template_id": prop_id,
        "query": template.format(row["subj"]),
        "gold": row["obj"],
        "alias": _alias(row),
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def cloze_query(row) -> str:
    """The cloze that stops exactly before the answer (== olmes ``doc_to_text``).

    This is the *query*, not a whole prompt: it is the tail of an n-shot prompt as
    much as it is the entirety of a 0-shot one.  Use it when you mean "this fact
    rendered as a cloze" (e.g. building a demonstration, or the query at the end of
    a few-shot block).

    When you specifically mean "the prompt with NOTHING conditioning the answer
    position" -- gold_prob, logit lens, hidden states -- call ``zero_shot_cloze``
    below instead.  Same string, but the name records which property you rely on,
    so a later change to few-shot prompting doesn't silently alter a measurement
    that was meant to stay unconditioned.
    """
    task = _olmes_task()
    if task is not None:
        return task.doc_to_text(process_doc(row))
    return process_doc(row)["query"]


def zero_shot_cloze(row) -> str:
    """The cloze with no demonstrations conditioning the answer position.

    Deliberately a separate name from ``cloze_query`` (same value): call sites that
    use THIS one depend on the prompt being unconditioned, not merely on it being
    the cloze.  See ``cloze_query``'s docstring.
    """
    return cloze_query(row)


def doc_target(row) -> str:
    """olmes doc_to_target: a leading space + gold answer."""
    return " " + str(row["obj"])


def render_fewshot_block(demos) -> str:
    """Render demonstrations into the prefix that precedes a fact's cloze.

    Layout transcribed from the fork's ``Task.fewshot_context`` (non-chat, no
    description), which builds::

        labeled_examples = "\\n\\n".join(doc_to_text(ex) + doc_to_target(ex)
                                        for ex in fewshotex) + "\\n\\n"
        prompt           = labeled_examples + doc_to_text(doc)

    For PopQACloze ``doc_to_text`` is ``doc["query"]`` -- the bare cloze, with no
    ``Q:``/``A:`` wrapper (that's the *other* task, ``popqa.py``) -- and
    ``doc_to_target`` is ``" " + gold``.  Going through ``cloze_query`` /
    ``doc_target`` means demo text comes from the real olmes task when olmes is
    installed, and from the transcribed cloze templates otherwise.

    ``demos`` is any iterable of row-likes (dicts or pandas Series).  Empty ->
    ``""``, which makes ``make_prompt`` collapse to the bare cloze.

    *Which* facts to demonstrate is not decided here -- see ``src.fewshot``, which
    deliberately does not use upstream's eval-set sampling.
    """
    demos = list(demos)
    if not demos:
        return ""
    return "\n\n".join(cloze_query(d) + doc_target(d) for d in demos) + "\n\n"


def make_prompt(row, prefix: str = "") -> str:
    """The primary recall prompt: demonstration block + this fact's cloze.

    With the default empty ``prefix`` this is exactly the cloze query --
    byte-identical to olmes ``doc_to_text`` and to what this function returned
    before few-shot support existed.  Build a prefix with
    ``src.fewshot.FewShot.build(...)`` and prefer ``FewShot.prompt(row)``.
    """
    return prefix + cloze_query(row)


def gen_kwargs() -> dict:
    """HuggingFace ``generate`` kwargs sourced from olmes generation_kwargs.
    Stop sequences are applied by truncation (see ``truncate_at_stop``)."""
    return {"max_new_tokens": GENERATION_KWARGS["max_gen_toks"],
            "do_sample": GENERATION_KWARGS["do_sample"]}


def truncate_at_stop(text: str) -> str:
    for s in GENERATION_KWARGS["stop_sequences"]:
        idx = text.find(s)
        if idx >= 0:
            text = text[:idx]
    return text


def answer_labels(row) -> list[str]:
    """The strings olmes accepts as this fact's answer: ``possible_answers`` with
    ``len > 3``.

    This is the *scoring* notion of "the answer" -- both ``is_excluded`` and
    ``is_correct`` below match against exactly this list, so anything that wants to
    reason about answer identity (e.g. ``src.fewshot``'s same-relation mode, which
    must not demonstrate a fact whose answer would be scored correct for the query)
    should go through here rather than re-deriving the rule.
    """
    return [l for l in process_doc(row)["alias"] if len(l) > 3]


def is_excluded(row) -> bool:
    """olmes drops examples whose answer already appears in the query."""
    doc = process_doc(row)
    labels = [l for l in doc["alias"] if len(l) > 3]
    q = doc["query"]
    return any(l in q or l.lower() in q or l.capitalize() in q for l in labels)


def is_correct(row, generation_text: str) -> bool:
    """Exact-match rule from PopQACloze.process_results (case-insensitive
    substring match of any len>3 label against the generation)."""
    task = _olmes_task()
    doc = process_doc(row)
    if task is not None:
        res = task.process_results(doc, [generation_text])
        return bool(res.get("exact_match", 0.0))
    labels = [l for l in doc["alias"] if len(l) > 3]
    pred = truncate_at_stop(generation_text)
    return any(l in pred or l.lower() in pred or l.capitalize() in pred
               for l in labels)
