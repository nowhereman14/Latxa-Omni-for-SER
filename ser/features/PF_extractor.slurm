#!/bin/bash
#SBATCH --job-name=acoustic_features_extraction
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/agarciam/acoustic_%j.out
#SBATCH --error=/scratch/agarciam/acoustic_%j.err

module load Miniforge3/24.11.3-2
export PATH="/home/agarciam/.conda/envs/llama-omni/bin:$PATH"

cd /scratch/agarciam/tfm/models/Latxa-Omni-Emotion
python -u PF_extractor.py