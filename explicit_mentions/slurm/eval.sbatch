#!/bin/bash
# Evaluate PopQA recall on converted checkpoints (or a published baseline) with
# the shared feature_extraction/eval_checkpoints.py (recall only, --no-internal).
# Submit FROM the explicit_mentions directory (log paths are relative):
#   sbatch slurm/eval.sbatch --model <repo>/data/hf/explicit_170M_1E \
#       --run-tag translated_corpus/explicit_170M_1E --prompts <repo>/data/artifacts/prompts/prompts_5shot_onlysameprop.parquet
#   sbatch slurm/eval.sbatch --model dhgottesman/LMEnt-170M-1E --steps 109672 \
#       --run-tag original_corpus/LMEnt_HF_models/Baseline/170M --prompts <same file>
# (or, instead of --prompts: --num-shots 5 --fewshot-mode only-same-prop)
# Outputs: <repo>/data/artifacts/recall/<run-tag>/recall_step{N}_*_recallonly.parquet
#SBATCH --job-name=eval_popqa
#SBATCH --partition=gpu-mol
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
#SBATCH --signal=USR1@120
#SBATCH --output=logs/eval_%A.out
#SBATCH --error=logs/eval_%A.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate esm_env

EM_DIR=/home/morg/NLP_2526b/shaharmayo/NLPFinalProject/explicit_mentions
cd "$EM_DIR"
python ../feature_extraction/eval_checkpoints.py --no-internal "$@"
