import json
import argparse
import time
import random
from collections import Counter
import torch
import whisper
from sklearn.metrics import classification_report, confusion_matrix
from ser.system_prompt import load_prompt
from omni_speech.conversation import conv_templates
from omni_speech.model.builder import create_model
from omni_speech.datasets.preprocess import tokenizer_speech_token
from ser.train.instruct_ft import load_manifest

LABELS = ["harridura", "haserrea", "neutroa", "poza", "tristura"]


def build_inference_prompt(conv_mode):
    conv = conv_templates[conv_mode].copy()
    conv.system = load_prompt()
    conv.append_message(conv.roles[0], "<speech>")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def prediction(audio_path, tokenizer, model, conv_mode, mel_size, device, use_mask):
    speech = whisper.load_audio(audio_path)
    speech = whisper.pad_or_trim(speech)
    speech_mel = whisper.log_mel_spectrogram(speech, n_mels=mel_size).permute(1, 0)

    speech_tensor = speech_mel.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
    speech_lengths = torch.LongTensor([speech_mel.shape[0]]).to(device)

    prompt = build_inference_prompt(conv_mode)
    input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)

    gen_kwargs = dict(
        speech=speech_tensor,
        speech_lengths=speech_lengths,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_new_tokens=10,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    if use_mask:
        # comportamiento anterior: con pad_token_id == <|eot_id|> esta mascara elimina
        # los <|eot_id|> del prompt (omni_speech_arch.py, linea 134)
        gen_kwargs["attention_mask"] = input_ids.ne(tokenizer.pad_token_id).to(device)

    with torch.inference_mode():
        outputs = model.generate(input_ids, **gen_kwargs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def mode_name(use_mask):
    return "with_mask" if use_mask else "without_mask"


def report(name, y_true, y_pred):
    invalid = [p for p in y_pred if p not in LABELS]
    print(f"\n===== {name} =====")
    print(f"Predicciones fuera de las 5 etiquetas: {len(invalid)} / {len(y_pred)}")
    if invalid:
        print("Ejemplos:", Counter(invalid).most_common(5))
    print(classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred, labels=sorted(set(y_true))))


def evaluation(args):
    print('Creating model...', flush=True)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer, model, context_len = create_model(
        model_path=args.model_path,
        model_base=args.model_base,
        is_lora=False,
        s2s=args.s2s,
        device=device)
    print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id,
          "| eos_token:", tokenizer.eos_token, tokenizer.eos_token_id, flush=True)
    model = model.to(device=device, dtype=torch.bfloat16)
    print(f"Model ready in {time.time()-t0:.1f}s", flush=True)
    model.eval()

    manifest = load_manifest(args.manifest_path)
    entries = [e for e in manifest if e["split"] == args.split]
    if args.limit:
        random.seed(42)
        random.shuffle(entries)
        entries = entries[:args.limit]
    print(f"{args.split}: {len(entries)}")

    modes = {"with": [True], "without": [False], "both": [True, False]}[args.mask_mode]
    y_true = []
    speakers = []
    preds = {m: [] for m in modes}

    for i, entry in enumerate(entries):
        y_true.append(entry["output"])
        speakers.append(entry["speaker"])
        for m in modes:
            preds[m].append(prediction(entry["input"], tokenizer, model,
                                       args.conv_mode, args.mel_size, device, m))
        if i % 100 == 0:
            print(f"Processed {i}/{len(entries)}", flush=True)

    for m in modes:
        out = f"{args.split}_predictions_{args.tag}_{mode_name(m)}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"y_true": y_true, "y_pred": preds[m], "speakers": speakers},
                      f, ensure_ascii=False, indent=2)
        print(f"Guardado: {out}")
        report(mode_name(m), y_true, preds[m])

    if args.mask_mode == "both":
        a, b = preds[True], preds[False]
        changed = [(i, a[i], b[i], y_true[i]) for i in range(len(a)) if a[i] != b[i]]
        acc_a = sum(x == t for x, t in zip(a, y_true)) / len(a)
        acc_b = sum(x == t for x, t in zip(b, y_true)) / len(b)
        print("\n===== COMPARACION with_mask vs without_mask =====")
        print(f"Accuracy with_mask:    {acc_a:.4f}")
        print(f"Accuracy without_mask: {acc_b:.4f}")
        print(f"Predicciones que cambian: {len(changed)} / {len(a)} ({100*len(changed)/len(a):.1f}%)")
        print("Cambios (with -> without), los mas frecuentes:")
        for (x, y), n in Counter((c[1], c[2]) for c in changed).most_common(10):
            print(f"  {x} -> {y}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="saves/final")
    parser.add_argument("--model-base", type=str, default="weights/Latxa-3.1-8B-Omni")
    parser.add_argument("--manifest-path", type=str, default="manifests/manifest_1.jsonl")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conv_mode", type=str, default="llama_3")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=False)
    parser.add_argument("--mask-mode", type=str, default="without", choices=["with", "without", "both"])
    parser.add_argument("--tag", type=str, default="run")
    args = parser.parse_args()
    evaluation(args)