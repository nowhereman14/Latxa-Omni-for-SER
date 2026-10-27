#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=Omni-Stage1 # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=3-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/Omni-Stage1-%j.out.log
#SBATCH --error=.slurm/Omni-Stage1-%j.err.log
#SBATCH --chdir=/home/andoni.sudupe/probak/llama-omni-eng-pa

export SSL_CERT_FILE=~/cacert.pem
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
# python omni_speech/train/stage1.py \
#     --model-path HiTZ/Latxa-Llama-3.1-8B-Instruct \
#     --question-file /scratch/andoni.sudupe/Instruct_S2S_eu/data.json \
#     --num-chunks 1 \
#     --chunk-idx 0 \
#     --temperature 0 \
#     --conv-mode llama_3 \
#     --output-folder $SLURM_JOB_ID \

MODEL_PATH=Llama-3.2-1B-Instruct
# TRAIN-FILE=/scratch/andoni.sudupe/Instruct_S2S_eu/hf_16k
TRAIN_FILE=yuekai/InstructS2S-200K
BATCH_SIZE=32
TRAIN_BATCH_SIZE=16


python "/shared/home/andoni.sudupe/probak/llama-omni-eng-pa/omni_speech/train/stage1.py" \
    --model-path $MODEL_PATH \
    --train-file $TRAIN_FILE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $(($BATCH_SIZE / ($TRAIN_BATCH_SIZE * $GPUS_PER_NODE))) \
    --output_dir saves/$MODEL_PATH/$SLURM_JOB_ID\_reproduce \
    --run_name Proba_$MODEL_PATH\_$SLURM_JOB_ID\_reproduce