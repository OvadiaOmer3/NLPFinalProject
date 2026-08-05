"""Convert OLMo-core training checkpoints to HuggingFace format.

Wraps the authors' OLMo-core script (examples/huggingface/convert_checkpoint_to_hf.py)
to produce the same layout the published LMEnt repos use -- one HF model per
step in stepN/ subfolders -- so eval_explicit.py can sweep them:

    <EXPLICIT_ROOT>/hf/<name>/step10000/{config.json, model.safetensors, tokenizer...}

Usage:
    python convert_checkpoints.py \
        --run-dir <EXPLICIT_ROOT>/runs/explicit_170M_1E/olmo2_170M_0.0003_32768_0.01_1 \
        --name explicit_170M_1E [--steps 10000 20000 ...]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

CONVERT_SCRIPT = (config.OLMO_CORE_PATH / "src" / "examples" / "huggingface"
                  / "convert_checkpoint_to_hf.py")


def ensure_local_tokenizer(out_root: Path) -> Path:
    """Materialize the dolma2 tokenizer locally so conversion works offline."""
    tok_dir = out_root / "_tokenizer"
    if not (tok_dir / "tokenizer_config.json").exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            config.TOKENIZER_ID, subfolder=config.TOKENIZER_SUBFOLDER, use_fast=True)
        tok_dir.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(tok_dir)
    return tok_dir


# transformers 5.x (the training env) writes tokenizer_class="TokenizersBackend"
# plus a few v5-only keys. transformers 4.x (the eval env) rejects that class,
# so rewrite the metadata to the portable equivalent. Only this JSON differs --
# tokenizer.json itself is version-independent.
V5_ONLY_KEYS = ("backend", "is_local", "local_files_only")


def normalize_tokenizer_config(model_dir: Path):
    cfg_path = model_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    changed = False
    if cfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        changed = True
    # Without this the fast tokenizer also returns token_type_ids, which
    # model.generate() rejects with "model_kwargs are not used by the model".
    # The published LMEnt models avoid it via GPT2Tokenizer, whose input names
    # are exactly these two.
    if cfg.get("model_input_names") != ["input_ids", "attention_mask"]:
        cfg["model_input_names"] = ["input_ids", "attention_mask"]
        changed = True
    for k in V5_ONLY_KEYS:
        if k in cfg:
            del cfg[k]
            changed = True
    if changed:
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"[convert] normalised {cfg_path.name} for transformers 4.x")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="the OLMo-core save_folder containing stepN/ checkpoints "
                         "(runs/<run-name>/<model>_<lr>_<bs>_<wd>_<dur>)")
    ap.add_argument("--name", required=True, help="output name under hf/")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="steps to convert (default: every stepN found)")
    ap.add_argument("--max-sequence-length", type=int, default=None,
                    help="value for the HF config's max_position_embeddings. "
                         "The upstream converter reads dataset.sequence_length, "
                         "which is null for the VSL dataset we train with, so it "
                         "must be supplied. Default: read "
                         "dataset.max_sequence_length from the checkpoint's own "
                         "config.json (2048).")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_root = config.HF_DIR / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    tok_dir = ensure_local_tokenizer(out_root)

    step_dirs = sorted(
        (p for p in run_dir.glob("step*") if re.fullmatch(r"step\d+", p.name)),
        key=lambda p: int(p.name[4:]))
    if args.steps is not None:
        wanted = set(args.steps)
        step_dirs = [p for p in step_dirs if int(p.name[4:]) in wanted]
    if not step_dirs:
        sys.exit(f"no step directories found in {run_dir}")

    env_path = f"{config.OLMO_CORE_PATH / 'src'}"
    for step_dir in step_dirs:
        seq_len = args.max_sequence_length
        if seq_len is None:
            # The VSL dataset leaves `sequence_length` null and puts the real
            # value in `max_sequence_length`; upstream only looks at the former
            # and dies with "Missing or invalid sequence length: -1".
            try:
                ck_cfg = json.loads((step_dir / "config.json").read_text())
                seq_len = ck_cfg.get("dataset", {}).get("max_sequence_length")
            except Exception:
                seq_len = None
            if not seq_len or seq_len <= 0:
                seq_len = 2048
        out_dir = out_root / step_dir.name
        if args.skip_existing and (out_dir / "config.json").exists():
            print(f"[skip] {out_dir} exists")
            continue
        print(f"[convert] {step_dir} -> {out_dir}")
        subprocess.run(
            [sys.executable, str(CONVERT_SCRIPT),
             "-i", str(step_dir), "-o", str(out_dir), "-t", str(tok_dir),
             "-s", str(seq_len)],
            check=True,
            env={**__import__("os").environ, "PYTHONPATH": env_path},
        )
        normalize_tokenizer_config(out_dir)
    print(f"done: {out_root}")


if __name__ == "__main__":
    main()
