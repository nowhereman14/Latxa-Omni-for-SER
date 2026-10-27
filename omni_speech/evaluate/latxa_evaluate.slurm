#!/usr/bin/env bash
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --job-name=evaluate_omni # Name of the process
#SBATCH --gres=gpu:2 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=128GB # RAM memory needed (8-16GB)
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/latxa_evaluate_omni-%j.out.log
#SBATCH --error=.slurm/latxa_evaluate_omni-%j.err.log

python "/dipc/asudupe/Latxa-Omni/omni_speech/evaluate/latxa_evaluate.py"