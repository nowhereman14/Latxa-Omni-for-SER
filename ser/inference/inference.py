import os
import torch
import torchaudio
from omni_speech.conversation import conv_templates, SeparatorStyle
from omni_speech.model.builder import load_pretrained_model
from omni_speech.datasets.preprocess import tokenizer_speech_token
import whisper
import numpy as np
from speechbrain.inference.vocoders import UnitHIFIGAN
from system_prompt import load_prompt 

DATA_ROOT = os.environ.get("SER_DATA_ROOT", "/scratch/agarciam/tfm/data")

def ctc_postprocess(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = [v for v in deduplicated_toks if v != blank] #官方493 222
    hyp = " ".join(list(map(str, hyp))) #1918 547
    return hyp

model_path = "Latxa-3.1-8B-Omni"
model_base = None
is_lora = False
s2s = True
mel_size = 128
conv_mode = 'llama_3'
tokenizer, model, context_len = load_pretrained_model(model_path, model_base, is_lora=is_lora, s2s=s2s)

hifigan = UnitHIFIGAN.from_hparams(source="HiFiGAN-Basque-Maider-Antton", run_opts={"device":'cuda'})

qs = load_prompt()
speech_file_1 = os.path.join(DATA_ROOT, "TTS_DB", "jaione_eu_angry", "wav", "JIEA0396.wav")
speech_file_2 = os.path.join(DATA_ROOT, "TTS_DB", "pello2004_eu_sad", "wav", "TBT303.wav")
speech_file_3 = os.path.join(DATA_ROOT, "TTS_DB", "karolina_eu_happy", "wav", "TBP408.wav")
audio_1 = whisper.load_audio(speech_file_1)
audio_2 = whisper.load_audio(speech_file_2)
audio_3 = whisper.load_audio(speech_file_3)

conv = conv_templates[conv_mode].copy()
conv.system = qs

conv.append_message(conv.roles[0], f"<speech>")
conv.append_message(conv.roles[1], "haserrea")

conv.append_message(conv.roles[0], "<speech>")
conv.append_message(conv.roles[1], "tristura")

conv.append_message(conv.roles[0], "<speech>")
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()
print("=== GENERATED PROMPT ===")
print(prompt)
print("=== END OF THE PROMPT ===")

audios = [audio_1, audio_2, audio_3]
speech_features_list = []
speech_lengths_list = []

for aud in audios:
    speech_padded = whisper.pad_or_trim(aud)
    speech_mel = whisper.log_mel_spectrogram(speech_padded, n_mels=mel_size).permute(1, 0)

    speech_features_list.append(speech_mel)
    speech_lengths_list.append(speech_mel.shape[0])

speech_tensor = torch.stack(speech_features_list).to(dtype=torch.float16, device='cuda', non_blocking=True)
speech_lengths = torch.LongTensor(speech_lengths_list).to(device='cuda', non_blocking=True)

input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt').to(device='cuda', non_blocking=True)

input_ids = input_ids.unsqueeze(0)
speech_tensors = speech_tensor.unsqueeze(0).flatten(0, 1)
speech_lengths = speech_lengths.squeeze()

print("=== SHAPES DEBUG ===")
print("speech_tensor (antes de unsqueeze/flatten):", speech_tensor.shape)
print("speech_tensors (final, lo que se pasa al modelo):", speech_tensors.shape)
print("speech_lengths_list (original):", speech_lengths_list)
print("speech_lengths (final, lo que se pasa al modelo):", speech_lengths.shape, speech_lengths)
print("input_ids shape:", input_ids.shape)
print("Numero de <speech> en el prompt:", prompt.count("<speech>"))
print("=== END SHAPES DEBUG ===")

temperature = 0
top_p = None
num_beams = 1
max_new_tokens = 512

with torch.inference_mode():
    outputs = model.generate(
        input_ids,
        speech=speech_tensors,
        speech_lengths=speech_lengths,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=128004,
        streaming_unit_gen=True,
 
    )
output_ids, output_units = outputs

print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip())
output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)
output_units = torch.tensor([int(x) for x in output_units.split()], dtype=torch.long)
answer = hifigan.decode_unit(output_units.unsqueeze(-1), torch.tensor(np.load('HiFiGAN-Basque-Maider-Antton/speaker_embeddings/antton.npy')))
torchaudio.save("erantzuna.wav", answer.cpu(), sample_rate=16000)
