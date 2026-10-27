#!/bin/bash
#SBATCH --job-name=wav2vec_eval
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/agarciam/tfm/outputs/baselines/wav2vec/wav2vec_eval_5.out
#SBATCH --error=/scratch/agarciam/tfm/outputs/baselines/wav2vec/wav2vec_eval_5.err

module load Miniforge3/24.11.3-2
export PATH=/home/agarciam/.conda/envs/llama-omni/bin:$PATH
export HF_HOME=/scratch/agarciam/.cache/huggingface

cd /scratch/agarciam/tfm/baselines/wav2vec

python3 evaluation.py \
    --model-path /hitz_data/agarciam/saves_wav2vec/saves_wav2vec_5/final \
    --manifest-path /scratch/agarciam/tfm/manifests/manifest_5.jsonl \
    --split test \
    --output /scratch/agarciam/tfm/outputs/baselines/wav2vec/wav2vec_test_fold5.json