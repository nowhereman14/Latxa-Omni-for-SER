import argparse
import torch
from omni_speech.conversation import conv_templates
from omni_speech.model.builder import load_pretrained_model
from omni_speech.datasets.preprocess import tokenizer_speech_token
import whisper
from datasets import load_from_disk
import numpy as np
from speechbrain.inference.vocoders import UnitHIFIGAN
from omni_speech.utils import disable_torch_init
from transformers import pipeline
import os
from tqdm import tqdm
from jiwer import wer, cer
import re
# from gradio_client import Client, file
import string
import json
import torchaudio

def ctc_postprocess(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = [v for v in deduplicated_toks if v != blank] #官方493 222
    hyp = " ".join(list(map(str, hyp))) #1918 547
    return hyp


# client = Client("HiTZ/Demo_Basque_ASR")
# result = client.predict(
# 		file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),	# filepath in 'Audio' Audio component
# 		api_name="/predict"
# )
# print(result)



def eval_model(args):
    # Model initialization
    disable_torch_init()

    model_base = None
    is_lora = False
    s2s = True
    mel_size = 128
    conv_mode = 'llama_3'
    
    tokenizer, model, context_len = load_pretrained_model(args.model_path, model_base, is_lora=is_lora, s2s=s2s)
    hifigan = UnitHIFIGAN.from_hparams(source="/scratch/asudupe/models/hifigan/spk_5/", run_opts={"device":'cuda'})

    dataset = load_from_disk('/scratch/asudupe/datasets/VoiceAssistant-400K_eu/')
    dataset = dataset['test']
    # dataset = dataset.select(range(10))

    # hf_model = "HiTZ/whisper-large-v3-eu"

    # pipe = pipeline(
    #     task="automatic-speech-recognition",
    #     model=hf_model,
    #     device="cuda"
    # )

    # wer_by_split = {} 
    # cer_by_split = {} 
    erantzunak = {}

    for id, example in enumerate(tqdm(dataset)):

        if os.path.exists(f"/scratch/asudupe/wer_audioak/{id}.wav"):
            continue

        split_name = example['split_name'] 

        speech_file = example['question_audio']
        speech_file = os.path.join('/scratch/asudupe/datasets/VoiceAssistant-400K_eu', speech_file)
        qs = "<speech>\nPlease directly answer the questions in the user's speech."
        
        speech_loaded = whisper.load_audio(speech_file)

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        speech = whisper.pad_or_trim(speech_loaded)
        speech = whisper.log_mel_spectrogram(speech, n_mels=mel_size).permute(1, 0)

        input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt')
        speech_length = torch.LongTensor([speech.shape[0]])

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech.to(dtype=torch.float16, device='cuda', non_blocking=True)
        speech_length = speech_length.to(device='cuda', non_blocking=True)

        input_ids = input_ids.unsqueeze(0)
        speech_tensors = speech_tensor.unsqueeze(0)
        speech_lengths = speech_length.unsqueeze(0)

        temperature = None
        top_p = None
        num_beams = 1
        max_new_tokens = 512

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                speech=speech_tensors,
                speech_lengths=speech_lengths,
                do_sample= False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=128004,
                streaming_unit_gen=True,
            )
        
        output_ids, output_units = outputs

        out_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        out_text = out_text.lower()

        translator = str.maketrans('', '', string.punctuation)
        out_text = out_text.translate(translator)

        output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)
        
        output_units_tensor = torch.tensor([int(x) for x in output_units.split()], dtype=torch.long)
        answer = hifigan.decode_unit(output_units_tensor.unsqueeze(-1), torch.tensor(np.load('/scratch/asudupe/models/hifigan/sonora_2/alex.npy')))


        
        # result = pipe(answer.squeeze(0).cpu().numpy(), return_timestamps=True)
        # transcribed_text = re.sub(r'\.(?!\s)', '. ', result["text"])

        # transcribed_text = transcribed_text.lower()

        # translator = str.maketrans('', '', string.punctuation)
        # transcribed_text = transcribed_text.translate(translator)

        erantzunak['split'] = split_name
        erantzunak['out_text'] = out_text
        erantzunak['transcribed_text'] = None
        erantzunak['id'] = id

        torchaudio.save(f"/scratch/asudupe/wer_audioak/{id}.wav", answer.cpu(), sample_rate=16000)
    
    with open('results/wer_data.json', 'w') as f:
        json.dump(erantzunak, f)

        # Calculate WER
        # wer_error = wer(out_text, transcribed_text)
        # if split_name not in wer_by_split:
        #     wer_by_split[split_name] = []
        # wer_by_split[split_name].append(wer_error)

        # cer_error = cer(out_text, transcribed_text)
        # if split_name not in cer_by_split:
        #     cer_by_split[split_name] = []
        # cer_by_split[split_name].append(cer_error)


    # print("\n" + "="*70)
    # print("WER ETA CER EMAITZAK SPLIT BAKOITZEKO NORMALIZATUTA")
    # print("="*70)
    
    # total_wer_errors = []
    # total_cer_errors = []
    
    # for split, w_errors in wer_by_split.items():
    #     avg_wer = sum(w_errors) / len(w_errors)
        
    #     c_errors = cer_by_split.get(split, [])
    #     if c_errors:
    #         avg_cer = sum(c_errors) / len(c_errors)
    #     else:
    #         avg_cer = 0.0

    #     print(f"Split: {split:<20} | Samples: {len(w_errors):<5} | WER: {avg_wer:.4f} | CER: {avg_cer:.4f}")
        
    #     total_wer_errors.extend(w_errors)
    #     total_cer_errors.extend(c_errors)

    # if total_wer_errors:
    #     global_wer = sum(total_wer_errors) / len(total_wer_errors)
        
    #     global_cer = 0.0
    #     if total_cer_errors:
    #         global_cer = sum(total_cer_errors) / len(total_cer_errors)

    #     print("-" * 70)
    #     print(f"Global WER: {global_wer:.4f}")
    #     print(f"Global CER: {global_cer:.4f}")
    # print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    eval_model(args)
