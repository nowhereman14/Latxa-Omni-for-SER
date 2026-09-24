#!/bin/bash
#SBATCH --job-name=LatxaOmni_SER_evaluation
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4 # Number of CPU cores
#SBATCH --mem-per-gpu=128GB
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/agarciam/eval_fold_4.out
#SBATCH --error=/scratch/agarciam/eval_fold_4.err

module load Miniforge3/24.11.3-2
export PATH="/home/agarciam/.conda/envs/llama-omni/bin:$PATH"

export SSL_CERT_FILE=~/cacert.pem
export CUDA_DEVICE_MAX_CONNECTIONS=1
export USE_DEEPSPEED=0

GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)

which python
which deepspeed

cd /scratch/agarciam/tfm/models/Latxa-Omni-Emotion
#python -u evaluation.py --model-path Latxa-3.1-8B-Omni --model-base Latxa-3.1-8B-Omni --split test
#python -u silence_experiment.py --model-path saves_full_ft/final --model-base Latxa-3.1-8B-Omni --split test --limit 500
python -u evaluation.py --model-path saves_full_ft_4/final --manifest-path "manifest_4.jsonl" --split test