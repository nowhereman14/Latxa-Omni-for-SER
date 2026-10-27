import os
import json
from gradio_client import Client, file
from tqdm import tqdm

# 1. Configuration
AUDIO_FOLDER = '/scratch/asudupe/wer_audioak/'
JSON_PATH = '/home/asudupe/Latxa-Omni/results/evaluation_results.json' # Update this to your actual filename
client = Client("HiTZ/Demo_Basque_ASR")

# 2. Load your existing JSON data
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 3. Process each audio file
# We assume 'data' is a list of dictionaries
for entry in tqdm(data):
    audio_id = entry.get('id')  # Adjust key name if it's not exactly 'id'
    filename = f"{audio_id}.wav"
    filepath = os.path.join(AUDIO_FOLDER, filename)

    # Check if the file actually exists before calling the API
    if os.path.exists(filepath):
        try:
            print(f"Transcribing {filename}...")
            result = client.predict(
                file(filepath),
                api_name="/predict"
            )
            
            # Update the specific entry in memory
            entry['response_asr'] = result
            
            # 3. Save the entire file immediately after each success
            with open('/home/asudupe/Latxa-Omni/results/evaluation_transcribed.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"File {filename} not found.")

print("All available audios have been processed and saved.")