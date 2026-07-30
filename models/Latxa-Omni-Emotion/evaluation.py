import json
import argparse
import time
import random
import torch
import whisper
from sklearn.metrics import classification_report, confusion_matrix
from system_prompt import load_prompt
from omni_speech.conversation import conv_templates
from omni_speech.model.builder import create_model
from omni_speech.datasets.preprocess import tokenizer_speech_token
from instruct_ft import load_manifest

def build_inference_prompt(conv_mode):
    conv = conv_templates[conv_mode].copy()
    conv.system = load_prompt()
    conv.append_message(conv.roles[0], "<speech>")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def prediction(entry, tokenizer, model, conv_mode, mel_size, device):
    speech = whisper.load_audio(entry)
    speech = whisper.pad_or_trim(speech)
    speech_mel = whisper.log_mel_spectrogram(speech, n_mels=mel_size).permute(1, 0)

    speech_tensor = speech_mel.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
    speech_lengths = torch.LongTensor([speech_mel.shape[0]]).to(device)

    prompt = build_inference_prompt(conv_mode)
    input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            speech=speech_tensor,
            speech_lengths=speech_lengths,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=10,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return prediction

def evaluation(args):
    print('Creating model...', flush=True)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer, model, context_len = create_model(
        model_path= args.model_path,
        model_base = args.model_base,
        is_lora= False,
        s2s= args.s2s,
        device= device)
    print("¿CUDA available?:", torch.cuda.is_available(), flush=True)
    print("Current device:", torch.cuda.current_device(), flush=True)
    print("GPU name:", torch.cuda.get_device_name(0), flush=True)
    model = model.to(device=device, dtype=torch.bfloat16)
    print(f"Model already moved to GPU succesfully in {time.time()-t0:.1f}s!", flush=True)
    model.eval()
    
    manifest = load_manifest(args.manifest_path)
    entries = [e for e in manifest if e["split"] == args.split]
    if args.limit:
        random.seed(42)
        random.shuffle(entries)
        entries = entries[:args.limit]
    print(f"{args.split}: {len(entries)}")
    
    y_true, y_pred, speakers = [], [], []
    for i, entry in enumerate(entries):
        pred = prediction(entry["input"], tokenizer, model, args.conv_mode, args.mel_size, device)
        y_true.append(entry["output"])
        y_pred.append(pred)
        speakers.append(entry["speaker"])
        if i % 100 == 0:
            print(f"Processed {i}/{len(entries)}")

    results = {"y_true": y_true, "y_pred": y_pred, "speakers": speakers}
    output_filename = f"{args.split}_predictions.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred, labels=sorted(set(y_true))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="saves/final")
    parser.add_argument("--model-base", type=str, default="Latxa-3.1-8B-Omni")
    parser.add_argument("--manifest-path", type=str, default="manifest.jsonl")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conv_mode", type=str, default="llama_3")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=False)
    args = parser.parse_args()
    evaluation(args)

#python evaluation.py --model-path saves_r8_adapter/checkpoint-150 --split val --limit 200