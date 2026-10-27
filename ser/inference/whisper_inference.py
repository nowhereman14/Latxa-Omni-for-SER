import os
import types
import json
import math
import numpy as np
import torch
import whisper
from omni_speech.model.speech_encoder.builder import build_speech_encoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_MELS = 128

model_config = types.SimpleNamespace(
    speech_encoder_type="whisper",
    speech_encoder="weights/zuazo-whisper-large-v3-eu.pt",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

print("Loading Whisper-eu encoder ...", flush=True)
encoder = build_speech_encoder(model_config)
encoder = encoder.to(device)
encoder.eval()


def load_manifest(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def extract_whisper_embedding(audio_path):
    audio = whisper.load_audio(audio_path)
    n_samples = min(len(audio), whisper.audio.N_SAMPLES)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=N_MELS).to(device)
    mel = mel.unsqueeze(0)  # (1, n_mels, n_frames)

    with torch.no_grad():
        hidden_states = encoder(mel)  # (1, 1500, n_state)
        n_frames = max(1, math.ceil(n_samples / whisper.audio.N_SAMPLES_PER_TOKEN))  # 16 kHz -> 1 frame del encoder cada 320 muestras (hop 160 del mel x stride 2 de la conv)
        embedding = hidden_states[0, :n_frames].mean(dim=0) 

    return embedding.cpu().numpy()


def build_split(entries, split_name):
    embeddings, labels, speakers = [], [], []
    for i, entry in enumerate(entries):
        embeddings.append(extract_whisper_embedding(entry["input"]))
        labels.append(entry["output"])
        speakers.append(entry["speaker"])
        if i % 100 == 0:
            print(f"Processed {i}/{len(entries)}", flush=True)

    np.save(f"whisper_{split_name}_embeddings.npy", np.array(embeddings))
    np.save(f"whisper_{split_name}_labels.npy", np.array(labels))
    np.save(f"whisper_{split_name}_speakers.npy", np.array(speakers))
    print(f"Arrays saved successfully!", flush=True)


manifest = load_manifest(os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "models", "Latxa-Omni-Emotion", "manifests_cv_ser", "manifest_1.jsonl"))
print("Generating embeddings ...", flush=True)
build_split([e for e in manifest if e["split"] == "train"], "train")
build_split([e for e in manifest if e["split"] == "val"], "val")
build_split([e for e in manifest if e["split"] == "test"], "test")