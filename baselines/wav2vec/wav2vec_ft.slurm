#!/bin/bash
#SBATCH --job-name=wav2vec_ft
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4 # Number of CPU cores
#SBATCH --mem-per-gpu=64GB # RAM memory needed
#SBATCH --time=24:00:00
#SBATCH --output=outputs/baselines/wav2vec/wav2vec_ft_5.out
#SBATCH --error=outputs/baselines/wav2vec/wav2vec_ft_5.err

module load Miniforge3/24.11.3-2
export PATH="/home/agarciam/.conda/envs/llama-omni/bin:$PATH"
export HF_HOME=/scratch/agarciam/.cache/huggingface

cd /scratch/agarciam/tfm
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 baselines/wav2vec/wav2vec_ft.py --output_dir /hitz_data/agarciam/wav2vec_new/saves_wav2vec_5