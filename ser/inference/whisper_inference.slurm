#!/bin/bash
#SBATCH --job-name=whisper_inference
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4 # Number of CPU cores
#SBATCH --mem-per-gpu=32GB # RAM memory needed
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/agarciam/whisper_inference_%j.out
#SBATCH --error=/scratch/agarciam/whisper_inference_%j.err

module load Miniforge3/24.11.3-2
export PATH="/home/agarciam/.conda/envs/llama-omni/bin:$PATH"

cd /scratch/agarciam/tfm/models/Latxa-Omni-Emotion
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 whisper_inference.py