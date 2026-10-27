#!/usr/bin/env bash
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --constraint=a100-sxm4
#SBATCH --job-name=Latxa-Omni-Stage1 # Name of the process
#SBATCH --gres=gpu:4 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=64GB # RAM memory needed (8-16GB)
#SBATCH --time=4-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/Latxa-Omni-Stage1.out.log
#SBATCH --error=.slurm/Latxa-Omni-Stage1.err.log
#SBATCH --chdir=/home/asudupe/Latxa-Omni

export SSL_CERT_FILE=~/cacert.pem
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export WANDB_PROJECT='latxa-omni'

GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
GPUS_PER_NODE=4
# python omni_speech/train/stage1.py \
#     --model-path HiTZ/Latxa-Llama-3.1-8B-Instruct \
#     --question-file /scratch/andoni.sudupe/Instruct_S2S_eu/data.json \
#     --num-chunks 1 \
#     --chunk-idx 0 \
#     --temperature 0 \
#     --conv-mode llama_3 \
#     --output-folder $SLURM_JOB_ID \
MODEL_PATH=/scratch/asudupe/models/Latxa-Llama-3.1-8B-Instruct
MODEL_BASE=HiTZ/Latxa-Llama-3.1-8B-Instruct
MODEL_NAME=Latxa-Llama-3.1-8B-Instruct
# TRAIN_FILE=Ansu/Instruct_S2S_eu
TRAIN_FILE=/scratch/asudupe/datasets/VoiceAssistant-400K_eu
DATA_ROOT=/scratch/asudupe/datasets/VoiceAssistant-400K_eu
EPOCHS=4

BATCH_SIZE=64
TRAIN_BATCH_SIZE=16

torchrun --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_endpoint=localhost:29400 \
    "/home/asudupe/Latxa-Omni/omni_speech/train/stage1.py" \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --train-file $TRAIN_FILE \
    --data-root $DATA_ROOT \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $(($BATCH_SIZE / ($TRAIN_BATCH_SIZE * $GPUS_PER_NODE))) \
    --output_dir /scratch/asudupe/checkpoints/$MODEL_NAME/stage1/$SLURM_JOB_ID \
    --num_train_epochs $EPOCHS \
    --run_name $MODEL_NAME\_VoiceAssistant\_$SLURM_JOB_ID\_whisper_eu \
    
