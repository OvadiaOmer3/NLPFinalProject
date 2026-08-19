"""Train an LMEnt-style model on a (rewritten) corpus.

Adapted from the LMEnt authors' OLMo-core fork, `src/examples/kas/train.py`,
with three changes:
  * imports olmo_core from the fork's checkout (config.OLMO_CORE_PATH) so this
    directory stays self-contained;
  * WandB and the downstream evaluator are controlled by the config json
    ("enabled" keys) instead of being hardcoded on;
  * a top-level "load_path" config key initializes the model from an existing
    OLMo-core checkpoint dir (e.g. phase 1's stepN folder) when the run's own
    save_folder has no checkpoint yet -- this implements the two-phase
    schedule "train on the original corpus, then continue on the explicit one".

Launch with torchrun (see slurm/train.sbatch):
    torchrun --nproc-per-node=1 train_explicit.py <config.json>
Configs are produced by make_train_config.py.
"""
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as em_config

sys.path.insert(0, str(em_config.OLMO_CORE_PATH / "src"))

# The fork's olmo_core/train/callbacks/wandb.py does `os.environ["WANDB_API_KEY"] = ""`
# at import time, silently destroying whatever key the environment provided.
# Capture it before that import and restore it afterwards.
_WANDB_API_KEY = os.environ.get("WANDB_API_KEY") or None

import numpy as np
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True

from olmo_core.config import Config, DType
from olmo_core.data import (
    KASDataCollator,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import VSLCurriculumConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    ConsoleLoggerCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    ProfilerCallback,
    SchedulerCallback,
    WandBCallback,
)
from olmo_core.train.common import Duration
from olmo_core.utils import get_default_device, seed_all

if _WANDB_API_KEY:                      # undo the import-time clobbering
    os.environ["WANDB_API_KEY"] = _WANDB_API_KEY


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    optim: AdamWConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_config(config: Dict[str, Any]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.dolma2()

    model_name = config["model"]
    peak_lr = config["optim"]["lr"]
    weight_decay = config["optim"]["weight_decay"]

    ds = config["dataset"]
    vsl = ds["vsl_curriculum"]

    dl = config["data_loader"]
    tr = config["trainer"]
    cbs = tr["callbacks"]

    md = tr["max_duration"]
    if md["unit"] == "epochs":
        max_duration = Duration.epochs(md["value"])
    elif md["unit"] == "steps":
        max_duration = Duration.steps(md["value"])
    else:
        raise ValueError(f"Unsupported max_duration unit: {md['unit']}")

    save_folder = Path(
        tr["save_folder"],
        f"{model_name}_{peak_lr}_{dl['global_batch_size']}_{weight_decay}_{max_duration.value}",
    )
    save_folder.mkdir(parents=True, exist_ok=True)

    match model_name:
        case "olmo2_170M":
            model_builder = TransformerConfig.olmo2_170M
        case "olmo2_190M":
            model_builder = TransformerConfig.olmo2_190M
        case "olmo2_600M":
            model_builder = TransformerConfig.olmo2_600M
        case "olmo2_1B":
            model_builder = TransformerConfig.olmo2_1B
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    model_config = model_builder(
        vocab_size=tokenizer_config.padded_vocab_size(),
        compile=True,
        fused_ops=False,
        use_flash=False,
        # FSDP, measured 20% faster than DDP here (0.2512 vs 0.3025 s/step,
        # matched over steps 2000-8000 of original_170M_1E on 4x L40S).
        # Counting collectives suggests otherwise -- FSDP does ~21 per step to
        # DDP's one -- but the bytes are equal (~1.49 GB/GPU/step each way),
        # and FSDP wins on two things DDP can't match here: params are natively
        # bf16 so there is no autocast casting on every op, and the optimizer
        # step is sharded (each rank runs AdamW over 1/4 of the params instead
        # of all of them). Sharding also becomes a hard requirement at
        # olmo2_1B, where AdamW state alone is ~21 GB against 44 GB of device
        # memory -- so this is the one strategy that works at every size.
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )

    optim_config = AdamWConfig(
        lr=peak_lr,
        group_overrides=[
            OptimGroupOverride(
                params=["embeddings.weight"], opts=dict(weight_decay=weight_decay)
            )
        ],
    )

    dataset_config = NumpyDatasetConfig.glob(
        *ds["paths"],
        name=ds["name"],
        max_sequence_length=ds["max_sequence_length"],
        min_sequence_length=ds["min_sequence_length"],
        vsl_curriculum=VSLCurriculumConfig(
            name=vsl["name"], num_cycles=vsl["num_cycles"], balanced=vsl["balanced"]
        ),
        tokenizer=tokenizer_config,
        work_dir=ds["work_dir"],
        include_instance_metadata=ds["include_instance_metadata"],
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=dl["global_batch_size"],
        seed=dl["seed"],
        num_workers=dl["num_workers"],
        prefetch_factor=dl["prefetch_factor"],
    )

    trainer_config = (
        TrainerConfig(
            save_folder=str(save_folder),
            rank_microbatch_size=tr["rank_microbatch_size"],
            save_overwrite=tr["save_overwrite"],
            metrics_collect_interval=tr["metrics_collect_interval"],
            cancel_check_interval=tr["cancel_check_interval"],
            max_duration=max_duration,
            # No autocast_precision: FSDP's param_dtype already gives bf16
            # compute, and stacking autocast on top only adds casts. Side
            # effect: SpeedMonitor only reports MFU when autocast_precision is
            # bfloat16, so MFU is absent from the logs under FSDP.
            load_key_mapping={
                "lm_head.w_out.weight": "w_out.weight",
                "lm_head.norm.weight": "norm.weight",
            },
        )
        .with_callback(
            "lr_scheduler",
            SchedulerCallback(
                scheduler=CosWithWarmup(warmup_steps=cbs["lr_scheduler"]["warmup_steps"])
            ),
        )
        # Register explicitly: otherwise the trainer auto-adds one with
        # log_interval=1, which prints a "[step=N/total]" line for every one of
        # ~114k steps regardless of metrics_collect_interval.
        .with_callback(
            "console_logger",
            ConsoleLoggerCallback(
                log_interval=cbs.get("console_logger", {}).get("log_interval", 100),
                metrics_log_interval=cbs.get("console_logger", {}).get(
                    "log_interval", 100),
            ),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "grad_clipper",
            GradClipperCallback(max_grad_norm=cbs["grad_clipper"]["max_grad_norm"]),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=cbs["checkpointer"]["save_interval"],
                ephemeral_save_interval=cbs["checkpointer"]["ephemeral_save_interval"],
                save_async=cbs["checkpointer"]["save_async"],
                pre_train_checkpoint=cbs["checkpointer"].get(
                    "pre_train_checkpoint", None),
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=cbs["wandb"].get("name", str(save_folder)),
                project=cbs["wandb"].get("project"),
                entity=cbs["wandb"].get("entity"),
                group=cbs["wandb"].get("group"),
                tags=cbs["wandb"].get("tags"),
                notes=cbs["wandb"].get("notes"),
                cancel_check_interval=cbs["wandb"]["cancel_check_interval"],
                enabled=cbs["wandb"].get("enabled", False),
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
    )
    if cbs.get("downstream_evaluator", {}).get("enabled", False):
        trainer_config = trainer_config.with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=cbs["downstream_evaluator"]["tasks"],
                tokenizer=tokenizer_config,
                eval_interval=cbs["downstream_evaluator"]["eval_interval"],
            ),
        )

    return ExperimentConfig(
        model=model_config,
        optim=optim_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        trainer=trainer_config,
    )


def main(config_filepath: str):
    with open(config_filepath, "r") as f:
        config_dict = json.load(f)

    load_path = config_dict.pop("load_path", None)
    config = build_config(config_dict)
    print(config, flush=True)

    seed_all(config.init_seed)
    set_random_seeds(config.init_seed)

    device = get_default_device()
    world_mesh = config.model.build_mesh(device=device)

    model = config.model.build(
        init_device="meta",
        device=device,
        max_seq_len=config.dataset.sequence_length,
        mesh=world_mesh,
    )
    optim = config.optim.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset,
        collator=KASDataCollator(
            pad_token_id=dataset.pad_token_id,
            rank_batch_size=config.trainer.rank_microbatch_size,
        ),
        mesh=world_mesh,
    )

    trainer = config.trainer.build(model, optim, data_loader, mesh=world_mesh)

    # Two-phase schedule: start from an existing checkpoint (phase 1's weights)
    # unless this run already has its own checkpoint to resume from. A fresh
    # LR schedule / dataloader state is used for the new corpus.
    if load_path is not None and not trainer.maybe_load_checkpoint(trainer.save_folder):
        print(f"[phase-2] initializing model weights from {load_path}", flush=True)
        trainer.load_checkpoint(
            load_path, load_optimizer_state=False, load_trainer_state=False
        )

    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


if __name__ == "__main__":
    config_filepath = sys.argv[1]
    prepare_training_environment()
    try:
        main(config_filepath)
    finally:
        teardown_training_environment()
