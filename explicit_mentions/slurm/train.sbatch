#!/bin/bash
# Train on a (rewritten) corpus with OLMo-core.
# First generate a config: python make_train_config.py --run-name ... --corpus ...
# Then, from the explicit_mentions directory:
#   sbatch slurm/train.sbatch <EXPLICIT_ROOT>/runs/<run-name>/config.json
#
# Uses the lment_train conda env (see README "Environment"). The 24h killable
# limit is shorter than a full epoch; just resubmit the same command -- the
# trainer auto-resumes from the latest checkpoint in its save_folder.
#
# GPU COUNT: OLMo-core requires global_batch_size % (rank_microbatch_size *
# gpus) == 0. With the paper's 32768/8192 that allows only 1, 2 or 4 GPUs --
# hence --gpus=4. To use 8, regenerate the config with
# --gpus 8 --rank-microbatch-size 4096 (same effective batch, more accumulation).
# torchrun below picks up whatever slurm actually granted.
#SBATCH --job-name=train_explicit
#SBATCH --partition=gpu-h100-killable
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200000
#SBATCH --signal=USR1@120
#SBATCH --output=logs/train_%A.out
#SBATCH --error=logs/train_%A.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lment_train_cu126

# W&B: `wandb login` once on the login node writes ~/.netrc (mode 600); read the
# key from there so it is never stored in this repo or echoed into a log.
if [ -z "$WANDB_API_KEY" ] && [ -f "$HOME/.netrc" ]; then
    export WANDB_API_KEY=$(python -c "import netrc,sys;a=netrc.netrc().authenticators('api.wandb.ai');print(a[2] if a else '')" 2>/dev/null)
fi
[ -n "$WANDB_API_KEY" ] && echo "[wandb] API key found" || echo "[wandb] no API key -- run wandb login, or leave --wandb off"

# NCCL peer-to-peer is BROKEN on n-801 -- for EVERY GPU pair, not just across
# sockets. Do not turn this off. Two failure modes, both from
# slurm/nccl_test.sbatch:
#   * hang: the first collective never returns (killed jobs 707979/708003,
#     diagnosed in job 710271).
#   * SILENT CORRUPTION, which is worse: job 761891 (4 GPUs, 0-3) and job
#     761982 (2 GPUs, 4-5 -- a PIX pair on one bridge in one NUMA node) both
#     passed the 16-element broadcast and then returned 1.0 from a 1M-element
#     all_reduce that should have returned the world size. Gradients are never
#     averaged, NCCL reports no error, and training completes normally with a
#     quietly wrong model.
# Small messages take a host-staged path and look fine; large ones use direct
# peer mappings and are the broken part -- so a quick smoke test that only
# broadcasts will wrongly appear healthy. Likely PCIe ACS or IOMMU config;
# fixing it needs a BIOS change or pcie_acs_override, i.e. cluster admins.
# Re-test ONLY via slurm/nccl_test.sbatch, which verifies collective *values*.
# Never by running training with NCCL_P2P_DISABLE=0 -- that yields no error,
# just bad science.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}
echo "[nccl] NCCL_DEBUG=$NCCL_DEBUG P2P_DISABLE=${NCCL_P2P_DISABLE:-unset} SHM_DISABLE=${NCCL_SHM_DISABLE:-unset}"

EM_DIR=/home/morg/NLP_2526b/shaharmayo/NLPFinalProject/explicit_mentions
cd "$EM_DIR"

CONFIG=$1
if [ -z "$CONFIG" ]; then echo "usage: sbatch slurm/train.sbatch <config.json>"; exit 1; fi

# How many GPUs did slurm actually give us? NOT `nvidia-smi -L` -- that lists
# every GPU on the node regardless of the allocation, so on a shared node it
# over-counts and the extra ranks die with "CUDA error: invalid device ordinal".
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NGPU=$SLURM_GPUS_ON_NODE
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NGPU=$(nvidia-smi -L | wc -l)
fi
echo "[gpus] allocated=$NGPU  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/[gpus] /'

# OLMo-core requires global_batch_size % (rank_microbatch_size * ngpu) == 0.
# Check it here so a mismatch fails in seconds with a clear message instead of
# after model init.
python - "$CONFIG" "$NGPU" <<'PYCHECK'
import json, sys
cfg = json.load(open(sys.argv[1])); n = int(sys.argv[2])
gb = cfg["data_loader"]["global_batch_size"]
rb = cfg["trainer"]["rank_microbatch_size"]
if gb % (rb * n):
    ok = [k for k in (1, 2, 4, 8, 16) if gb % (rb * k) == 0]
    sys.exit(f"FATAL: global_batch_size {gb} is not divisible by "
             f"rank_microbatch_size {rb} x {n} GPUs.\n"
             f"  valid GPU counts here: {ok}\n"
             f"  or regenerate the config with "
             f"--gpus {n} --rank-microbatch-size {gb // n}")
print(f"[batch] {n} GPUs x {rb} tokens, {gb // (rb * n)} accumulation step(s) "
      f"-> {gb} tokens/optimizer step")
PYCHECK
[ $? -ne 0 ] && exit 1

torchrun --nproc-per-node="$NGPU" train_explicit.py "$CONFIG"
