#!/bin/bash
#SBATCH --job-name=inference_test
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/agarciam/inference_%j.out
#SBATCH --error=/scratch/agarciam/inference_%j.err

export PATH=/home/agarciam/.conda/envs/llama-omni/bin:$PATH

echo "=== DEBUG INFO ==="
which python
python --version
python -c "import torchaudio; print(torchaudio.__version__)"
echo "=== END DEBUG ==="

cd /scratch/agarciam/tfm/models/Latxa-Omni-Emotion
python inference.py
