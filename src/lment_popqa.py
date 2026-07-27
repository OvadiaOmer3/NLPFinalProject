"""Single home for the olmes (LMEnt eval-harness) coupling.

Everything about *how the paper builds prompts and scores answers* lives here so
the rest of the project imports one clean API and stays readable.

Prompt construction and scoring are single-sourced from LMEnt's olmes fork
(``dhgottesman/olmes``, task ``PopQACloze`` in
``oe_eval/tasks/oe_eval_tasks/popqa_cloze.py``):

  * cloze prompts (models are not instruction-tuned) keyed by ``prop_id``;
  * few-shot context (default 15-shot) with a FIXED seed so every checkpoint
    sees identical demonstrations;
  * exact-match scoring: labels = ``possible_answers`` with ``len > 3``, the
    example is *excluded* if a label is already in the query, otherwise correct
    iff any label substring-matches the generation (case-insensitive variants);
  * generation kwargs: ``max_gen_toks=15``, greedy, ``stop=["\\n\\n"]``.

If the olmes package is importable (submodule checked out + ``pip install -e
olmes``) we defer to the real ``PopQACloze`` task -- that path is authoritative.
Otherwise we use the faithful fallback below, transcribed from the fork, so the
prompt/scoring-parity checks in the plan run without installing olmes.

# VERIFY at checkout: the exact olmes symbols (registry key, method signatures)
# against the on-disk fork -- these were read from the repo over the web.
"""
from __future__ import annotations

import functools
import json
import random

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
def zero_shot_cloze(row) -> str:
    """The unconditioned cloze that stops exactly before the answer (== olmes
    doc_to_text).  Used for gold_prob / logit-lens / hidden-states."""
    task = _olmes_task()
    if task is not None:
        return task.doc_to_text(process_doc(row))
    return process_doc(row)["query"]


def doc_target(row) -> str:
    """olmes doc_to_target: a leading space + gold answer."""
    return " " + str(row["obj"])


def make_prompt(row, num_shots: int | None = None, seed: int | None = None) -> str:
    """The primary recall prompt.

    The paper evaluates knowledge zero-shot (``config.NUM_SHOTS = 0``), and for
    zero-shot the prompt is exactly the cloze query -- verified byte-identical to
    olmes ``fewshot_context``/``doc_to_text`` (base model, non-chat).

    Few-shot (``num_shots > 0``, only for reproducing the OLMES *benchmark* table)
    is delegated to the real olmes task, which applies the demonstration
    alias-safety filtering we would otherwise have to re-implement.  We do NOT
    approximate it locally: if olmes isn't installed, we raise rather than emit a
    subtly-different prompt.
    """
    num_shots = config.NUM_SHOTS if num_shots is None else num_shots
    seed = config.FEWSHOT_SEED if seed is None else seed

    if num_shots <= 0:
        return zero_shot_cloze(row)          # exact zero-shot cloze == olmes

    task = _olmes_task()
    if task is None:
        raise RuntimeError(
            f"num_shots={num_shots} requires the olmes package (authoritative "
            "few-shot context). Install it (pip install -e LMEnt/olmes) or set "
            "config.NUM_SHOTS = 0 for the paper's zero-shot evaluation."
        )
    return task.fewshot_context(process_doc(row),
                                num_fewshot=num_shots,
                                rnd=random.Random(seed))


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
