import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from instruct_ft import load_manifest

def extract_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    intensity = snd.to_intensity()
    pitch = snd.to_pitch()

    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values != 0]

    if len(f0_values) == 0:
        return None

    return {
        "f0mean": f0_values.mean(),
        "f0min": f0_values.min(),
        "f0max": f0_values.max(),
        "f0range": f0_values.max() - f0_values.min(),
        "f0sd": f0_values.std(),
        "dBmean": intensity.values.mean(),
        "duration": snd.duration,
    }

manifest = load_manifest('manifest.jsonl')
#test_entries = [e for e in manifest if e["split"] == "test"]

rows = []
for i, entry in enumerate(manifest):
    try:
        features = extract_features(entry["input"])
        if features is None:
                continue
        features["emotion"] = entry["output"]
        features["speaker"] = entry["speaker"]
        features["audio_id"] = entry["input"]
        rows.append(features)
    except Exception as e:
        print(f"Error processing {entry['input']}: {e}")

    if i % 500 == 0:
        print(f"Processed {i}/{len(manifest)}")

df = pd.DataFrame(rows)
df.to_csv("acoustic_features_full_corpus.csv", index=False)
print(f"\nSaved with {len(df)} out of {len(manifest)} rows")

print("\n=== Results per emotion ===")
print(df.groupby("emotion")[["f0mean", "f0range", "f0sd", "dBmean"]].mean())