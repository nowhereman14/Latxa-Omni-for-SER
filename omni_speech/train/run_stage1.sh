#!/bin/bash

VOCODER_CKPT=vocoder/g_00500000
VOCODER_CFG=vocoder/config.json

export PYTHONPATH=$(pwd)
export WANDB_PROJECT='llama-omni'
export WANDB_LOG_MODEL='true'
export WANDB_WATCH='false'
export CUDA_VISIBLE_DEVICES='1'
export MODEL_PATH='Llama-3.1-8B-Omni'

python omni_speech/train/stage1.py \
    --model-path Llama-3.1-8B-Omni \
    --train-file speechocean/train.json \
    --valid-file speechocean/dev.json \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --eval_accumulation_steps 4 \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
    --input_type mel \
    --mel_size 128 \
    --num_train_epochs 5 \
    --output_dir /work/twsmwhk715/saves/stage1/omni-5e \
    --run_name "omni-5e-s1" \

