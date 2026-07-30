#!/bin/bash
#SBATCH --job-name=LatxaOmni_SER_evaluation_on_train
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4 # Number of CPU cores
#SBATCH --mem-per-gpu=128GB
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/agarciam/train_eval_%j.out
#SBATCH --error=/scratch/agarciam/train_eval_%j.err

module load Miniforge3/24.11.3-2
export PATH="/home/agarciam/.conda/envs/llama-omni/bin:$PATH"

export SSL_CERT_FILE=~/cacert.pem
export CUDA_DEVICE_MAX_CONNECTIONS=1
export USE_DEEPSPEED=0

GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)

which python
which deepspeed

cd /scratch/agarciam/tfm/models/Latxa-Omni-Emotion
python -u evaluation.py --model-path saves_full_ft/checkpoint-1050  --split train --limit 500