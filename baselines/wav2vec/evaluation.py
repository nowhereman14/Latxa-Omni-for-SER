import json
import argparse
import random
import time
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

SR = 16000


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class EvalDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        audio, _ = librosa.load(e["input"], sr=SR)
        return audio, e["output"], e["speaker"]


def make_collate(feature_extractor):
    def collate(batch):
        audios, labels, speakers = zip(*batch)
        feats = feature_extractor(
            list(audios), sampling_rate=SR, padding=True, return_tensors="pt"
        )
        return feats, list(labels), list(speakers)
    return collate


def evaluation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...", flush=True)
    t0 = time.time()
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_path)
    model = Wav2Vec2BertForSequenceClassification.from_pretrained(args.model_path)
    model.to(device).eval()
    id2label = model.config.id2label
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    manifest = load_manifest(args.manifest_path)
    entries = [e for e in manifest if e["split"] == args.split]
    if args.limit:
        random.seed(42)
        random.shuffle(entries)
        entries = entries[:args.limit]
    print(f"{args.split}: {len(entries)}", flush=True)

    loader = DataLoader(
        EvalDataset(entries),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(feature_extractor),
    )

    y_true, y_pred, speakers = [], [], []
    with torch.inference_mode():
        for i, (feats, labels, spk) in enumerate(loader):
            feats = {k: v.to(device) for k, v in feats.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = model(**feats).logits
            preds = logits.argmax(dim=-1).tolist()
            y_pred.extend(id2label[p] for p in preds)
            y_true.extend(labels)
            speakers.extend(spk)
            if i % 20 == 0:
                print(f"Processed {len(y_true)}/{len(entries)}", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"y_true": y_true, "y_pred": y_pred, "speakers": speakers},
                  f, ensure_ascii=False, indent=2)

    labels_sorted = sorted(set(y_true))
    print(classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred, labels=labels_sorted))
    print("Labels order:", labels_sorted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Carpeta con el modelo guardado, p. ej. /hitz_data/agarciam/saves_wav2vec/final")
    parser.add_argument("--manifest-path", type=str,
                        default="/scratch/agarciam/tfm/manifests/manifest_1.jsonl")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="wav2vec_test_predictions.json")
    args = parser.parse_args()
    evaluation(args)