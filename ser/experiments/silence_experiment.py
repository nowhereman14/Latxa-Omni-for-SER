import json
import argparse
import time
import random
import numpy as np
from collections import Counter
import torch
import whisper
from sklearn.metrics import classification_report, confusion_matrix
from omni_speech.conversation import conv_templates
from omni_speech.model.builder import create_model
from omni_speech.datasets.preprocess import tokenizer_speech_token
from instruct_ft import load_manifest
from evaluation import build_inference_prompt, prediction

def silence_experiment(args):
    print('Creating model...', flush=True)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer, model, context_len = create_model(
            model_path= args.model_path,
            model_base = args.model_base,
            is_lora= False,
            s2s= args.s2s,
            device= device)
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

    results = []
    for i, entry in enumerate(entries):
        pred_real = prediction(entry["input"], tokenizer, model, args.conv_mode, args.mel_size, device)

        silence_audio = np.zeros(16000 * 5, dtype=np.float32)
        speech = whisper.pad_or_trim(silence_audio)
        speech_mel = whisper.log_mel_spectrogram(speech, n_mels=args.mel_size).permute(1, 0)
        '''print("Shape:", speech_mel.shape)
        print("Min:", speech_mel.min().item(), "Max:", speech_mel.max().item())
        print("Unique values (first 10 ones):", torch.unique(speech_mel)[:10])'''
        speech_tensor = speech_mel.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
        speech_lengths = torch.LongTensor([speech_mel.shape[0]]).to(device)
        prompt = build_inference_prompt(args.conv_mode)
        input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids, attention_mask=attention_mask,
                speech=speech_tensor, speech_lengths=speech_lengths,
                do_sample=False, temperature=None, top_p=None,
                max_new_tokens=10, use_cache=True, pad_token_id=tokenizer.eos_token_id,
            )
        pred_silence = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        results.append({"true": entry["output"], "speaker": entry["speaker"], "pred_real": pred_real, "pred_silence": pred_silence})
        if i % 100 == 0:
            print(f"Processed {i}/{len(entries)}")

    with open("silence_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_same = sum(1 for r in results if r["pred_real"] == r["pred_silence"])
    print(f"Iguales: {n_same}/{len(results)} ({100*n_same/len(results):.1f}%)")
    print("Dist. AUDIO REAL:", Counter(r["pred_real"] for r in results))
    print("Dist. SILENCIO:", Counter(r["pred_silence"] for r in results))

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
    silence_experiment(args)

    '''"To verify that the silence input did not introduce any spurious acoustic variation, we inspected the resulting log-mel spectrogram directly: it consists of a single constant value (-1.5) across all 3000×128 positions, confirming that the model received a maximally uninformative, contentless acoustic signal — not random noise."'''