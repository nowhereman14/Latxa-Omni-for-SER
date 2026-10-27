import pandas as pd

df = pd.read_csv("acoustic_features_full_corpus.csv")

for col in ["f0mean", "f0range", "f0sd", "dBmean"]:
    speaker_mean = df.groupby("speaker")[col].transform("mean")
    df[f"{col}_normalized"] = df[col] - speaker_mean

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print("\n=== Results per emotion (normalized by speaker) ===")
print(df.groupby("emotion")[["f0mean_normalized", "f0range_normalized", "f0sd_normalized", "dBmean_normalized"]].mean())