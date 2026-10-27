#!/bin/bash

ROOT=speechocean/speech_units

VOCODER_CKPT=vocoder/g_00500000
VOCODER_CFG=vocoder/config.json

# python omni_speech/infer/convert_jsonl_to_txt.py $ROOT/answer.json $ROOT/answer.unit
python fairseq/examples/speech_to_speech/generate_waveform_from_code.py \
    --in-code-file $ROOT/km_labels/valid_0_1.km \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
    --results-path $ROOT/unit2wav_valid/ --dur-prediction