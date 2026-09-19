#!/bin/bash
# Diagnose the NCCL hang that killed training (first 16-element BROADCAST timing
# out after 30 min). Runs nccl_smoke.py twice -- once with default transport,
# once with peer-to-peer disabled -- so a single ~6 minute job tells you which
# setting works on the node you were given.
#
# Submit FROM the explicit_mentions directory (log paths are relative):
#   sbatch slurm/nccl_test.sbatch          # 4 GPUs, matches the training run
#   sbatch --gpus=2 slurm/nccl_test.sbatch # or any other count
#SBATCH --job-name=nccl_test
#SBATCH --partition=gpu-mol
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --output=logs/nccl_%A.out
#SBATCH --error=logs/nccl_%A.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lment_train

EM_DIR=/home/morg/NLP_2526b/shaharmayo/NLPFinalProject/explicit_mentions
cd "$EM_DIR"

if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NGPU=$SLURM_GPUS_ON_NODE
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NGPU=$(nvidia-smi -L | wc -l)
fi
echo "node=$(hostname)  gpus=$NGPU  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader | sed 's/^/[gpu] /'
echo
echo "### GPU topology (which links NCCL can use) ###"
nvidia-smi topo -m 2>/dev/null || echo "  (nvidia-smi topo unavailable)"

echo
echo "############ TEST 1: default transport ############"
timeout 360 torchrun --nproc-per-node="$NGPU" nccl_smoke.py
T1=$?
echo "TEST 1 exit code: $T1"

echo
echo "############ TEST 2: NCCL_P2P_DISABLE=1 ############"
NCCL_P2P_DISABLE=1 timeout 360 torchrun --nproc-per-node="$NGPU" nccl_smoke.py
T2=$?
echo "TEST 2 exit code: $T2"

echo
echo "############ VERDICT ############"
if [ $T1 -eq 0 ]; then
    echo "default transport WORKS -> just resubmit training as-is;"
    echo "the earlier hang was transient or specific to that allocation."
elif [ $T2 -eq 0 ]; then
    echo "default transport FAILS but P2P-disabled WORKS -> submit training with:"
    echo "  NCCL_P2P_DISABLE=1 sbatch slurm/train.sbatch <config.json>"
else
    echo "BOTH FAIL -> multi-GPU NCCL is broken on this node."
    echo "Fall back to a single GPU (no collectives at all):"
    echo "  python make_train_config.py --run-name <name> --corpus explicit_all \\"
    echo "      --model olmo2_600M --gpus 1"
    echo "  sbatch --gpus=1 slurm/train.sbatch <config.json>"
fi
