import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import librosa
from transformers import AutoFeatureExtractor, Wav2Vec2BertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score


TAG_MAP = {
            'neutroa': 0,
            'poza': 1,
            'tristura': 2,
            'harridura': 3,
            'haserrea': 4
}
MODEL_NAME = "facebook/w2v-bert-2.0"

def load_manifest(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries

class W2VBertSERDataset(Dataset):
    def __init__(self, entries, feature_extractor, sampling_rate=16000):
        self.entries = entries
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def __getitem__(self, index):
        entry = self.entries[index]
        audio, sr = librosa.load(entry["input"], sr=self.sampling_rate)
        inputs = self.feature_extractor(audio, sampling_rate=self.sampling_rate, return_tensors="pt")
        return {
            "input_features": inputs["input_features"].squeeze(0),
            "labels": TAG_MAP[entry["output"]]
        }

    def __len__(self):
        return len(self.entries)

def collate_fn(batch):
    for i in range(len(batch)):
            batch[i]= batch[i].values()

    input_features, labels = zip(*batch)
    lengths = [f.shape[0] for f in input_features]
    input_features = pad_sequence(input_features, batch_first=True)
    T_max = input_features.shape[1]
    lengths = torch.tensor(lengths)
    attention_mask = (torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)).long()

    labels = torch.tensor(labels)
    return {
        "input_features": input_features,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }

def fine_tuning(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
        label2id=TAG_MAP,
        id2label={v: k for k, v in TAG_MAP.items()},
        use_weighted_layer_sum=True,
    ).to(device)

    manifest = load_manifest(args.manifest_path)
    train_entries = [e for e in manifest if e["split"] == "train"]
    val_entries = [e for e in manifest if e["split"] == "val"]
    test_entries = [e for e in manifest if e["split"] == "test"]

    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}, Test: {len(test_entries)}")

    train_ds = W2VBertSERDataset(train_entries, feature_extractor)
    val_ds = W2VBertSERDataset(val_entries, feature_extractor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy='steps',
        save_steps=150,
        eval_strategy='steps',
        eval_steps=150,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        save_total_limit=7,
        seed=3407,
        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        report_to="none"
    )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            callbacks= [EarlyStoppingCallback(early_stopping_patience=3)]
        )

    trainer.train(ignore_keys_for_eval=["hidden_states", "attentions"])

    trainer.save_model(os.path.join(args.output_dir, "final"))
    feature_extractor.save_pretrained(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=str, default="manifests/manifest_5.jsonl")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default='saves')
    args = parser.parse_args()
    fine_tuning(args)

    