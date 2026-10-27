#!/usr/bin/env bash
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --job-name=evaluate_omni # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=128GB # RAM memory needed (8-16GB)
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/evaluate_omni_wer-%j.out.log
#SBATCH --error=.slurm/evaluate_omni_wer-%j.err.log

MODEL_PATH=Ansu/Latxa-3.1-8B-Omni

python "/dipc/asudupe/Latxa-Omni/omni_speech/evaluate/evaluate_wer.py" \
    --model-path $MODEL_PATH