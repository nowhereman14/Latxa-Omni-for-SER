#!/bin/bash

ROOT=$1

VOCODER_CKPT=vocoder/g_00500000
VOCODER_CFG=vocoder/config.json

python omni_speech/infer/infer.py \
    --model-path /work/twsmwhk715/saves/stage2/8b-omni-10e/checkpoint-310/ \
    --question-file $ROOT/test_questions.json \
    --answer-file $ROOT/predictions-omni-10e/answer.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
    --input_type mel \
    --mel_size 128 \
    --s2s
python omni_speech/infer/convert_jsonl_to_txt.py $ROOT/predictions-omni-10e/answer.json $ROOT/predictions-omni-10e/answer.unit
python fairseq/examples/speech_to_speech/generate_waveform_from_code.py \
    --in-code-file $ROOT/predictions-omni-10e/answer.unit \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
    --results-path $ROOT/predictions-omni-10e/answer_wav/ --dur-prediction

