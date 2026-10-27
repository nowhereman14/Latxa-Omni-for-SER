ROOT=$1

VOCODER_CKPT=vocoder/g_00500000
VOCODER_CFG=vocoder/config.json

python omni_speech/infer/infer.py \
    --model-path /work/twsmwhk715/saves/8b-omni-10e-accu/checkpoint-125 \
    --question-file $ROOT/test.json \
    --answer-file $ROOT/predictions/8b-omni-10e-accu.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
    --input_type mel \
    --mel_size 128 \