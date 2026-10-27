#!/usr/bin/env bash
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --job-name=transcribe_nemo # Name of the process
#SBATCH --cpus-per-task=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem=128GB # RAM memory needed (8-16GB)
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/transcribe_nemo.out.log
#SBATCH --error=.slurm/transcribe_nemo.err.log

python "/dipc/asudupe/Latxa-Omni/omni_speech/evaluate/transcribe_nemo.py"