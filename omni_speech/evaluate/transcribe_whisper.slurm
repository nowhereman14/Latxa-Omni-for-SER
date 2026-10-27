#!/usr/bin/env bash
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --job-name=transcribe_whisper # Name of the process
#SBATCH --gres=gpu:1 # Number of CPU cores (2 is reasonable)
#SBATCH --cpus-per-task=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem=128GB # RAM memory needed (8-16GB)
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/transcribe_whisper.out.log
#SBATCH --error=.slurm/transcribe_whisper.err.log

python "/dipc/asudupe/Latxa-Omni/omni_speech/evaluate/transcribe_whisper.py"