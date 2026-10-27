#!/bin/bash
#SBATCH --job-name=LatxaOmni_SER_mask_ab
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=128GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/agarciam/tfm/outputs/logs/eval_mask_ab_fold1.out
#SBATCH --error=/scratch/agarciam/tfm/outputs/logs/eval_mask_ab_fold1.err

module load Miniforge3/24.11.3-2
export PATH="/home/agarciam/.conda/envs/llama-omni/bin:$PATH"

export SSL_CERT_FILE=~/cacert.pem
export CUDA_DEVICE_MAX_CONNECTIONS=1
export USE_DEEPSPEED=0

which python

cd /scratch/agarciam/tfm
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u ser/eval/evaluation_mask_ab.py \
  --model-path /hitz_data/agarciam/saves_adam_098/saves_full_ft/checkpoint-600 \
  --manifest-path manifests/manifest_1.jsonl \
  --split val \
  --limit 200 \
  --mask-mode both \
  --tag fold1_ckpt600