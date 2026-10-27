import os
import json
import numpy as np
from funasr import AutoModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Loading emotion2vec model ...", flush=True)
model = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf")

def load_manifest(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries

def extract_emotion_embedding(audio_path):
    result = model.generate(audio_path, granularity="utterance", extract_embedding=True)
    embedding = result[0]["feats"]
    return embedding

def build_split(entries, split_name):
    embeddings, labels, speakers = [], [], []
    for i, entry in enumerate(entries):
        embeddings.append(extract_emotion_embedding(entry["input"]))
        labels.append(entry["output"])
        speakers.append(entry["speaker"])
        if i % 100 == 0:
                    print(f"Processed {i}/{len(entries)}", flush=True)

    np.save(f"emo2vec_{split_name}_embeddings.npy", np.array(embeddings))
    np.save(f"emo2vec_{split_name}_labels.npy", np.array(labels))
    np.save(f"emo2vec_{split_name}_speakers.npy", np.array(speakers))
    print(f"Arrays saved successfully!", flush=True)

manifest = load_manifest(os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)),"models", "Latxa-Omni-Emotion", "manifests_cv_ser", "manifest_1.jsonl"))
print(f"Generating embeddings ...", flush=True)
build_split([e for e in manifest if e["split"] == "train"], "train")
build_split([e for e in manifest if e["split"] == "val"], "val")
build_split([e for e in manifest if e["split"] == "test"], "test")