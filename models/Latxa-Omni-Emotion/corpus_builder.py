import re
import os
import json

def parse_gaitu(folder_path, file_name):
    speaker = folder_path.rstrip('/').split('/')[-2].replace('_emocional', '')
    emotion = re.sub(r'^([A-Za-z]+).*', r'\1', file_name)
    return speaker, emotion

def parse_emoziak(folder_path, file_name):
    speaker = folder_path.rstrip('/').split('/')[-1]
    emotion = re.sub(r'^([A-Za-z]+)_.*', r'\1', file_name)
    return speaker, emotion

def parse_ttsdb(folder_path, _file_name):
    folder_name = folder_path.rstrip('/').split('/')[-2]
    parts = folder_name.split('_')
    speaker = parts[0]
    emotion = parts[-1] if len(parts) > 2 else "neutral"
    return speaker, emotion

TAG_MAP = {
            'angry': 'haserrea',
            'happy': 'poza',
            'sad': 'tristura',
            'surprised': 'harridura',
            'neutral': 'neutroa',
            'HAR': 'harridura',
            'HAS': 'haserrea',
            'POZ': 'poza',
            'TRI': 'tristura',
            'NEU': 'neutroa'
}

def get_tag(emotion):
    return TAG_MAP.get(emotion)

CORPUS_PARSERS = {
    'TTS_GAITU-DATA': parse_gaitu,
    'HiTZSpeechSynthesisEmozioak_Dataset': parse_emoziak,
    'TTS_DB': parse_ttsdb,
}

# --- Corpus folder configuration ---

# --- TTS_DB ---
        # TTS_DB folders included in this study.
        # Speakers like aintzane_eu, amaia_eu, andrea_eu, etc. are excluded:
        # no emotion annotation
TTSDB_BASE = "/scratch/agarciam/tfm/data/TTS_DB"
TTSDB_FOLDERS = [
"karolina_eu", "karolina_eu_angry",
"karolina_eu_happy", "karolina_eu_sad", "karolina_eu_surprised",
"pello2004_eu", "pello2004_eu_angry", "pello2004_eu_happy", "pello2004_eu_sad", "pello2004_eu_surprised",
"jaione_eu", "jaione_eu_angry", "jaione_eu_happy", "jaione_eu_sad",
"kepa_eu", "kepa_eu_angry", "kepa_eu_happy", "kepa_eu_sad", "amaia_eu", "inaki_eu"]

# --- EMOZIOAK ---
EMOZIOAK_BASE = "/scratch/agarciam/tfm/data/HiTZSpeechSynthesisEmozioak_Dataset"
EMOZIOAK_FOLDERS = ["Antton", "Maider"]

# --- GAITU ---
        # TTS_GAITU-DATA folders included in this study.
        # Speakers like jon, miren, nerea, etc. are excluded:
        # no emotion annotation
GAITU_BASE = "/scratch/agarciam/tfm/data/TTS_GAITU-DATA"
GAITU_FOLDERS = ["mikel_emocional", "estitxu_emocional"]

def build_entries(folder_path):
    entries = []
    corpus = folder_path.split('/')[5]
    parser = CORPUS_PARSERS.get(corpus)
    if parser is None:
        return entries
    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.wav'):
            continue
        speaker, raw_tag = parser(folder_path, file_name)
        emotion = get_tag(raw_tag)
        if emotion is None:
            continue
        full_path = os.path.join(folder_path, file_name)
        entries.append({"input": full_path, "output": emotion, "speaker": speaker})
    return entries

TRAIN_SPEAKERS = {"karolina", "jaione", "Antton", "estitxu", "mikel"}
VAL_SPEAKERS = {"pello2004"}
TEST_SPEAKERS = {"Maider", "kepa", "amaia","inaki"}

def assign_split(speaker):
    if speaker in TRAIN_SPEAKERS:
        return "train"
    elif speaker in VAL_SPEAKERS:
        return "val"
    elif speaker in TEST_SPEAKERS:
        return "test"
    return None

def save_manifest(entries, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__": 
    all_entries = []
    for folder in TTSDB_FOLDERS:
        full_folder_path = os.path.join(TTSDB_BASE, folder, "wav")
        all_entries += build_entries(full_folder_path)

    for folder in EMOZIOAK_FOLDERS:
        full_folder_path = os.path.join(EMOZIOAK_BASE, folder)
        all_entries += build_entries(full_folder_path)
    
    for folder in GAITU_FOLDERS:
        full_folder_path = os.path.join(GAITU_BASE, folder, "wav")
        all_entries += build_entries(full_folder_path)

    print(f"Total entries: {len(all_entries)}")
    
    for entry in all_entries:
        entry["split"] = assign_split(entry["speaker"])

    unassigned = [e for e in all_entries if e["split"] is None]
    if unassigned:
        print(f"WARNING: {len(unassigned)} entries with unknown speaker:")
        print(set(e["speaker"] for e in unassigned))

    save_manifest(all_entries, "manifest.jsonl")

    print(f"Total entries: {len(all_entries)}")
    for split in ["train", "val", "test"]:
        count = sum(1 for i in all_entries if i["split"] == split)
        print(f"  {split}: {count}")
        for emotion in ["haserrea", "poza", "tristura", "harridura", "neutroa"]:
            emo_count = sum(1 for e in all_entries if e["split"] == split and e["output"] == emotion)
            print(      f"{emotion}: {emo_count}")