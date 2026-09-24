#!/bin/bash
#SBATCH --job-name=Latxa_SER_instruct_ft
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4 # Number of CPU cores
#SBATCH --mem-per-gpu=128GB # RAM memory needed
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/agarciam/fold_ft_4.out
#SBATCH --error=/scratch/agarciam/fold_ft_4.err

module load Miniforge3/24.11.3-2
export PATH="/home/agarciam/.conda/envs/llama-omni/bin:$PATH"

export SSL_CERT_FILE=~/cacert.pem
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)

which python
which deepspeed

cd /scratch/agarciam/tfm/models/Latxa-Omni-Emotion
deepspeed --num_gpus=$GPUS_PER_NODE --master_port=29502 instruct_ft.py --manifest-path "manifest_4.jsonl" --num_train_epochs 3 --train_batch_size 2 --gradient_accumulation_steps 8 --output_dir saves_full_ft_4