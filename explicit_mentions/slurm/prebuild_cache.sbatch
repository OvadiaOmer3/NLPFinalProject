#!/bin/bash
# Build OLMo-core's dataset caches on CPU, before requesting GPUs.
# Submit FROM the explicit_mentions directory (log paths are relative):
#   sbatch slurm/prebuild_cache.sbatch <repo>/data/runs/<name>/config.json
# Needs many cores: _write_document_indices fans out one process per corpus part.
#SBATCH --job-name=prebuild_cache
#SBATCH --partition=cpu-killable
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000
#SBATCH --output=logs/prebuild_%A.out
#SBATCH --error=logs/prebuild_%A.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lment_train

EM_DIR=/home/morg/NLP_2526b/shaharmayo/NLPFinalProject/explicit_mentions
cd "$EM_DIR"

CONFIG=$1
if [ -z "$CONFIG" ]; then echo "usage: sbatch slurm/prebuild_cache.sbatch <config.json>"; exit 1; fi
python prebuild_dataset_cache.py "$CONFIG"
