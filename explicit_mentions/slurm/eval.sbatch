#!/bin/bash
# Evaluate PopQA recall on converted checkpoints (or a published baseline).
# From the explicit_mentions directory:
#   sbatch slurm/eval.sbatch --model <EXPLICIT_ROOT>/hf/explicit_170M_1E \
#       --run-tag explicit_170M_1E --num-shots 5 --fewshot-mode only-same-prop
#   sbatch slurm/eval.sbatch --model dhgottesman/LMEnt-170M-1E --steps 109672 \
#       --run-tag baseline_170M_1E --num-shots 5 --fewshot-mode only-same-prop
#SBATCH --job-name=eval_explicit
#SBATCH --partition=killable
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --constraint=l40s
#SBATCH --signal=USR1@120
#SBATCH --output=logs/eval_%A.out
#SBATCH --error=logs/eval_%A.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate esm_env

EM_DIR=/home/morg/NLP_2526b/shaharmayo/NLPFinalProject/explicit_mentions
cd "$EM_DIR"
python eval_explicit.py "$@"
