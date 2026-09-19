#!/bin/bash
# Rewrite one corpus part per array task (8 parts total).
# Submit FROM the explicit_mentions directory (log paths are relative):
#   cd .../NLPFinalProject/explicit_mentions && mkdir -p logs
#   sbatch slurm/rewrite_corpus.sbatch [extra build_explicit_corpus.py args...]
# e.g.
#   sbatch slurm/rewrite_corpus.sbatch --corpus-tag explicit_all
#   sbatch slurm/rewrite_corpus.sbatch --corpus-tag explicit_popqa --target-qids popqa
#   sbatch slurm/rewrite_corpus.sbatch --corpus-tag explicit_half --replace-fraction 0.5
#SBATCH --job-name=rewrite_corpus
#SBATCH --partition=cpu-killable
#SBATCH --time=4-00:00:00
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --output=logs/rewrite_%A_%a.out
#SBATCH --error=logs/rewrite_%A_%a.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate esm_env
export TOKENIZERS_PARALLELISM=true

EM_DIR=/home/morg/NLP_2526b/shaharmayo/NLPFinalProject/explicit_mentions
cd "$EM_DIR"
python build_explicit_corpus.py --part "${SLURM_ARRAY_TASK_ID}" "$@" 