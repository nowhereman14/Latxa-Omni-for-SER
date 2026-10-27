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
import string
import json
import time
# import google.generativeai as genai

# --- CONFIGURATION ---
# PLEASE SET YOUR GEMINI API KEY HERE OR IN ENVIRONMENT VARIABLES
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
OUTPUT_JSON_FILE = "/dipc/asudupe/Latxa-Omni/results/evaluation_results.json"

# if "YOUR_API_KEY_HERE" in GEMINI_API_KEY:
#     print("WARNING: Gemini API Key not found. Please set the GEMINI_API_KEY environment variable or edit the script.")

# genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini Model
# gemini_model = genai.GenerativeModel('gemini-3-pro') # Or gemini-pro / gemini-1.5-pro

def ctc_postprocess(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = [v for v in deduplicated_toks if v != blank] 
    hyp = " ".join(list(map(str, hyp))) 
    return hyp

def clean_text(text):
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def evaluate_with_gemini(instruction_text, response_text):
    """
    Sends the instruction and response to Gemini for critical evaluation.
    """
    prompt = f"""
I need your help to evaluate the performance of several models in a speech interaction scenario. The models are trained for Basque, so the texts will be in this language. The models receive the user’s speech input and respond with speech output. For evaluation purposes, both the user’s speech input and the model’s speech response have been transcribed into text using Automatic Speech Recognition (ASR).

Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response]. Please consider factors such as helpfulness, relevance, fluency, and suitability for speech interaction.

**IMPORTANT: Be extremely critical in your evaluation.**
* Check for **factual accuracy** closely. If the model states a scientific misconception as fact, penalize the score heavily (e.g., score it 1 or 2), even if the grammar is perfect.
* Look out for **hallucinations** or contradictions within the text.
* Ensure the terminology is precise.

Provide a single score on a scale from 1 to 5.

Below are the transcription of user’s instruction and models’ response:
### [Instruction]: {instruction_text}
### [Response]: {response_text}

After evaluating, please output the scores in JSON format: {{score: ...}}. You don’t need to provide any explanations.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        # Simple extraction of JSON from response (handling potential markdown code blocks)
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text_response)
        return result.get("score")
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def save_result(data, filename):
    """
    Appends a single result to a JSON list in a file. 
    Reads the file first to append correctly (not efficient for huge files, but safe).
    """
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                current_data = json.load(f)
            except json.JSONDecodeError:
                current_data = []
    else:
        current_data = []
    
    current_data.append(data)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(current_data, f, ensure_ascii=False, indent=4)

def eval_model(args):
    # Model initialization
    disable_torch_init()

    model_base = None
    is_lora = False
    s2s = True
    mel_size = 128
    conv_mode = 'llama_3'
    
    tokenizer, model, context_len = load_pretrained_model(args.model_path, model_base, is_lora=is_lora, s2s=s2s)
    # hifigan = UnitHIFIGAN.from_hparams(source="/scratch/asudupe/models/hifigan/spk_5/", run_opts={"device":'cuda'})

    dataset = load_from_disk('/scratch/asudupe/datasets/VoiceAssistant-400K_eu/')
    dataset = dataset['test']
    # dataset = dataset.select(range(10)) # Uncomment for testing

    # hf_model = "HiTZ/whisper-large-v3-eu"

    # pipe = pipeline(
    #     task="automatic-speech-recognition",
    #     model=hf_model,
    #     device="cuda"
    # )

    # wer_by_split = {} 
    # cer_by_split = {} 

    print(f"Starting evaluation... Results will be saved to {OUTPUT_JSON_FILE}")

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        split_name = example['split_name'] 
        speech_file_path = os.path.join('/scratch/asudupe/datasets/VoiceAssistant-400K_eu', example['question_audio'])
        
        # --- 1. Load User Audio ---
        qs = "<speech>\nPlease directly answer the questions in the user's speech."
        speech_loaded = whisper.load_audio(speech_file_path)
        instruction_text = example['question']

        # --- 2. Transcribe User Audio (Instruction) for Gemini ---
        # We use the same pipeline to transcribe the input to get the text for the prompt
        # try:
        #     user_asr_out = pipe(speech_loaded, return_timestamps=True)
        #     instruction_text = user_asr_out["text"]
        # except Exception as e:
        #     print(f"Error transcribing input audio: {e}")
        #     instruction_text = "[Error in transcription]"

        # --- 3. Generate Model Response ---
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

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                speech=speech_tensors,
                speech_lengths=speech_lengths,
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=128004,
                streaming_unit_gen=False,
            )
        
        output_ids, output_units = outputs
        # output_ids = outputs

        # Text generated by LLM (Ground Truth for Synthesis)
        out_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        out_text_clean = clean_text(out_text)

        # Audio Synthesis from Units
        # output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)
        # output_units_tensor = torch.tensor([int(x) for x in output_units.split()], dtype=torch.long)
        
        # Decode to Audio
        # answer_audio = hifigan.decode_unit(output_units_tensor.unsqueeze(-1), torch.tensor(np.load('/scratch/asudupe/models/hifigan/sonora_2/alex.npy')))
        
        # --- 4. Transcribe Model Response (ASR) ---
        # result = pipe(answer_audio.squeeze(0).cpu().numpy(), return_timestamps=True)
        # transcribed_text = re.sub(r'\.(?!\s)', '. ', result["text"])
        # transcribed_text_clean = clean_text(transcribed_text)

        # --- 5. Calculate Metrics (WER/CER) ---
        # wer_error = wer(out_text_clean, transcribed_text_clean)
        # cer_error = cer(out_text_clean, transcribed_text_clean)

        # if split_name not in wer_by_split:
        #     wer_by_split[split_name] = []
        # wer_by_split[split_name].append(wer_error)

        # if split_name not in cer_by_split:
        #     cer_by_split[split_name] = []
        # cer_by_split[split_name].append(cer_error)

        # --- 6. Gemini Evaluation ---
        # We compare the [Instruction] (User Input ASR) with the [Response] (Model Output ASR)
        gemini_score = evaluate_with_gemini(instruction_text, out_text)
        
        # --- 7. Save Result ---
        result_entry = {
            "id": i,
            "split": split_name,
            "instruction_asr": instruction_text,
            "response_text_llm": out_text,
            # "response_asr": transcribed_text,
            # "wer": wer_error,
            # "cer": cer_error,
            "gemini_score": None
        }
        
        save_result(result_entry, OUTPUT_JSON_FILE)
        
        # Optional: Sleep briefly to avoid hitting rate limits aggressively
        # time.sleep(1)

    # --- Summary Printing ---
    # print("\n" + "="*70)
    # print("SUMMARY RESULTS")
    # print("="*70)
    
    # total_wer_errors = []
    # total_cer_errors = []
    
    # for split, w_errors in wer_by_split.items():
    #     avg_wer = sum(w_errors) / len(w_errors)
    #     c_errors = cer_by_split.get(split, [])
    #     avg_cer = sum(c_errors) / len(c_errors) if c_errors else 0.0

    #     print(f"Split: {split:<20} | Samples: {len(w_errors):<5} | WER: {avg_wer:.4f} | CER: {avg_cer:.4f}")
        
    #     total_wer_errors.extend(w_errors)
    #     total_cer_errors.extend(c_errors)

    # if total_wer_errors:
    #     global_wer = sum(total_wer_errors) / len(total_wer_errors)
    #     global_cer = sum(total_cer_errors) / len(total_cer_errors) if total_cer_errors else 0.0

    #     print("-" * 70)
    #     print(f"Global WER: {global_wer:.4f}")
    #     print(f"Global CER: {global_cer:.4f}")
    
    # # Calculate Average Gemini Score
    # try:
    #     with open(OUTPUT_JSON_FILE, 'r') as f:
    #         saved_results = json.load(f)
    #         scores = [r['gemini_score'] for r in saved_results if r['gemini_score'] is not None]
    #         if scores:
    #             print(f"Average Gemini Score: {sum(scores)/len(scores):.2f}")
    # except:
    #     pass

    # print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()
    eval_model(args)