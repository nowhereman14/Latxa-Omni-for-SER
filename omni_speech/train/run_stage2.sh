#!/bin/bash
export PYTHONPATH=$(pwd)
export WANDB_PROJECT='llama-omni'
export WANDB_LOG_MODEL='true'
export WANDB_WATCH='false'
export MODEL_PATH='Llama-3.1-8B-Omni'
export CUDA_VISIBLE_DEVICES='1'

python omni_speech/train/stage2.py  \
    --model-path Llama-3.1-8B-Omni \
    --model-base Llama-3.1-8B-Omni \
    --train-question-file speechocean/stage2/train_questions.json \
    --train-answer-file speechocean/stage2/train_answers.json \
    --valid-question-file speechocean/stage2/valid_questions.json \
    --valid-answer-file speechocean/stage2/valid_answers.json \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --eval_accumulation_steps 4 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
    --input_type mel \
    --mel_size 128 \
    --num_train_epochs 10 \
    --output_dir /work/twsmwhk715/saves/stage2/8b-omni-10e \
    --run_name "8b-omni-10e-s2" \
    

