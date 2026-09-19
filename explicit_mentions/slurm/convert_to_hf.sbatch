#!/bin/bash
# Convert OLMo-core checkpoints to HF format (CPU job).
# Submit FROM the explicit_mentions directory (log paths are relative):
#   sbatch slurm/convert_to_hf.sbatch --run-dir <repo>/data/runs/<run>/olmo2_170M_... --name explicit_170M_1E
# -> <repo>/data/hf/<name>/stepN/
#SBATCH --job-name=convert_hf
#SBATCH --partition=cpu-killable
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --output=logs/convert_%A.out
#SBATCH --error=logs/convert_%A.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lment_train

EM_DIR=/home/morg/NLP_2526b/shaharmayo/NLPFinalProject/explicit_mentions
cd "$EM_DIR"
python convert_checkpoints.py "$@"
