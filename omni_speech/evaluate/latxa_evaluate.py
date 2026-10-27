import json
import os
import time
import argparse
import re
from json.decoder import JSONDecodeError
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "HiTZ/Latxa-Llama-3.1-70B-Instruct" 

# Adjust based on your GPU setup. 
# For a 70B model: 
# - TP=4 for 4x 24GB/40GB cards
# - TP=2 for 2x 80GB cards

def setup_pipeline():
    print(f"🚀 Loading {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Load model in 4-bit to save memory (fits in ~40-48GB VRAM)
    # If you have 2x A100s or similar, you can remove load_in_4bit=True
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",  # Automatically spreads model across GPUs
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Create generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=False
    )
    
    return pipe, tokenizer

def create_prompt(tokenizer, instruction, response_text):
    """
    Formats the input using the model's specific chat template.
    """
    
    prompt = f"""
I need your help to evaluate the performance of several models in a speech interaction scenario. The models are trained for Basque, so the texts will be in this language. The models receive the user's speech input and respond with text output. For evaluation purposes, the user's speech input have been transcribed into text using Automatic Speech Recognition (ASR).

Your task is to rate the model's responses based on the provided user input transcription [Instruction] and the model's output transcription [Response]. Please consider factors such as helpfulness, relevance, fluency, and suitability for speech interaction.

**IMPORTANT: Be extremely critical in your evaluation.**
* Check for **factual accuracy** closely. If the model states a scientific misconception as fact, penalize the score heavily (e.g., score it 1 or 2), even if the grammar is perfect.
* Look out for **hallucinations** or contradictions within the text.
* Ensure the terminology is precise.

Provide a single score on a scale from 1 to 5.

### SCORING RUBRIC (1-5 Scale):
- **5 (Excellent):** Perfect fluency, accurate facts, helpful, and natural tone. No errors.
- **4 (Good):** Very good response, but has minor nitpicks (e.g., slightly unnatural phrasing, a minor pronunciation error in text form, or slightly verbose). Completely accurate.
- **3 (Acceptable):** Understandable and helpful, but has noticeable issues (e.g., obvious anglicisms, robotic phrasing, or partially incomplete info).
- **2 (Poor):** Hard to follow, contains significant factual errors, or hallucinates information. 
- **1 (Bad):** Completely irrelevant, wrong language, dangerous, or nonsensical.

### INSTRUCTIONS:
1. Analyze the helpfulness, relevance, accuracy, and Basque fluency.
2. Identify any specific errors (hallucinations, grammar, tone).
3. Assign a score based STRICTLY on the rubric above. Use the full range (2, 3, 4) for intermediate quality.
4. Output your reasoning first, just a phrase or two, don't extend, then the final JSON {{score: ...}}.

    """
    user_content = f"""
### [Instruction]: {instruction}
### [Response]: {response_text}

"""

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content}
    ]

    # Apply chat template (e.g., adds <|start_header_id|>... etc.)
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return full_prompt

# def extract_score(text):
#     """
#     Extracts {score: X} or {"score": X} from the output text using Regex.
#     """
#     match = re.search(r'\{.*"score"\s*:\s*(\d+).*\}', text, re.DOTALL | re.IGNORECASE)
#     if match:
#         return int(match.group(1))
    
#     # Fallback for simple number output if model fails JSON
#     match_simple = re.search(r'\b([1-5])\b', text)
#     if match_simple:
#         return int(match_simple.group(1))
        
#     return None

def extract_score(text):
    """
    Extracts {score: X} or {"score": X} from the output text.
    Handles cases where the model provides reasoning text before the JSON.
    """
    # Look for the JSON pattern specifically at the end or embedded in text
    match = re.search(r'\{.*"score"\s*:\s*(\d+).*\}', text, re.DOTALL | re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Backup: Look for "Score: X" or "Rating: X" pattern if JSON fails
    match_text = re.search(r'(?:score|rating)\s*[:=]\s*(\d+)', text, re.IGNORECASE)
    if match_text:
        return int(match_text.group(1))

    return None

def load_json_safe(filepath):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (JSONDecodeError, Exception):
        return None

def save_result_incremental(output_file, result_item):
    current_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
        except JSONDecodeError:
            current_data = []
    
    current_data.append(result_item)
    
    temp_file = output_file + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(current_data, f, ensure_ascii=False, indent=4)
    os.replace(temp_file, output_file)

def watch_and_evaluate(args):
    # 1. Initialize Model
    pipe, tokenizer = setup_pipeline()

    print(f"👀 Watching '{args.input}'...")
    print(f"💾 Saving to '{args.output}'")

    processed_ids = set()
    if os.path.exists(args.output):
        data = load_json_safe(args.output)
        if data:
            processed_ids = {item.get('id') for item in data}
            print(f"✅ Resume: Found {len(processed_ids)} already processed items.")

    while True:
        try:
            input_data = load_json_safe(args.input)
            
            if input_data is None:
                time.sleep(1)
                continue

            # Find new items
            new_items = [item for item in input_data if item.get('id') not in processed_ids]

            new_items = new_items[3000:]
            if not new_items:
                time.sleep(5)
                continue

            print(f"\n🚀 Found {len(new_items)} new items. Processing...")

            for item in tqdm(reversed(new_items)):
                # Double check inside loop
                if item['id'] in processed_ids: continue

                instr = item.get('instruction_asr', item.get('instruction', ''))
                resp = item.get('response_text_llm', item.get('response', ''))
                
                # Format Prompt
                prompt = create_prompt(tokenizer, instr, resp)
                
                # Generate
                outputs = pipe(
                    prompt, 
                    max_new_tokens=512, 
                    do_sample=False, # Deterministic
                    temperature=0.0,
                    return_full_text=False
                )
                
                generated_text = outputs[0]['generated_text']
                score = extract_score(generated_text)
                
                # print(f"   ID {item['id']}: Score {score}")

                result_item = item.copy()
                result_item['latxa_score'] = score
                
                save_result_incremental(args.output, result_item)
                processed_ids.add(item['id'])
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/evaluation_results.json", help="Input JSON file")
    parser.add_argument("--output", type=str, default="results/latxa_scores_new_reversed.json", help="Output JSON file")
    
    args = parser.parse_args()
    watch_and_evaluate(args)