#!/bin/bash
#SBATCH --job-name=emotion2vec_inference
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4 # Number of CPU cores
#SBATCH --mem-per-gpu=28GB # RAM memory needed
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/agarciam/emotion2vec_inference_%j.out
#SBATCH --error=/scratch/agarciam/emotion2vec_inference_%j.err

module load Miniforge3/24.11.3-2
export PATH="/home/agarciam/.conda/envs/emotion2vec/bin:$PATH"

cd /scratch/agarciam/tfm/models/baseline
export PYTHONPATH="$PWD:$PYTHONPATH"
export MODELSCOPE_CACHE=/scratch/agarciam/.cache/modelscope

mkdir -p $MODELSCOPE_CACHE

python3 emotion2vec_inference.py