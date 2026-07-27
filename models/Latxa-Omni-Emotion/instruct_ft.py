import json
import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from peft import LoraConfig, get_peft_model
import whisper
from system_prompt import load_prompt
from omni_speech.conversation import conv_templates, SeparatorStyle
from omni_speech.model.builder import create_model
from omni_speech.datasets.preprocess import tokenizer_speech_token
from omni_speech.constants import IGNORE_INDEX
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_manifest(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries

class SERDataset(Dataset):
    def __init__(self, entries, tokenizer, conv_mode="llama_3", mel_size = 128):
        self.entries = entries
        self.tokenizer = tokenizer
        self.conv_mode = conv_mode
        self.mel_size = mel_size
   
    def __getitem__(self, index):
        entry = self.entries[index]
        speech_file = entry["input"]
        speech = whisper.load_audio(speech_file)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)
        
        conv = conv_templates[self.conv_mode].copy()
        conv.system = load_prompt()
        conv.append_message(conv.roles[0], "<speech>")
        conv.append_message(conv.roles[1], entry["output"])
        prompt = conv.get_prompt()

        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')
        target = input_ids.clone()

        sep = "<|start_header_id|>" + conv.roles[1] + "<|end_header_id|>\n\n"
        parts = prompt.split(sep)
        parts[0] += sep

        instruction_len = len(tokenizer_speech_token(parts[0], self.tokenizer)) - 1
        target[:instruction_len] = IGNORE_INDEX

        return dict(
            input_ids=input_ids,
            labels=target,
            speech=speech.to(torch.bfloat16),
            speech_lengths=torch.LongTensor([speech.shape[0]]))
    
    def __len__(self):
        return len(self.entries)

class EarlyEvalCallback(TrainerCallback):
    def __init__(self, early_eval_steps=(10, 25, 50, 75, 100)):
        self.early_eval_steps = set(early_eval_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.early_eval_steps:
            control.should_evaluate = True
        return control

def collate_fn(batch):
    for i in range(len(batch)):
        batch[i]= batch[i].values()
        
    input_ids,labels,speech_tensors,speech_lengths = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=128009)
    labels = pad_sequence(labels, batch_first=True, padding_value=128009)

    speech_tensors = torch.stack(speech_tensors, dim=0)
    speech_lengths = torch.stack(speech_lengths, dim=0)
    return {"input_ids":input_ids,"labels":labels, "speech":speech_tensors, "speech_lengths":speech_lengths}

def apply_lora(model, rank):
    config = LoraConfig(
        r = rank,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    #Unfreeze the Speech Adaptor besides LoRA
    
    for name, param in model.named_parameters():
        if "speech_projector" in name or "mm_projector" in name:  # nombres a confirmar con el listado de módulos
            param.requires_grad = True

    model.print_trainable_parameters()
    return model

def fine_tuning(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    '''from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Latxa-3.1-8B-Omni", use_fast=False)'''
    tokenizer, model, context_len = create_model(
        model_path= args.model_path,
        model_base = args.model_base,
        is_lora= False,
        s2s= args.s2s,
        device= device)

    print(tokenizer.eos_token, tokenizer.eos_token_id)
    print(tokenizer.pad_token, tokenizer.pad_token_id)

    model = apply_lora(model, rank = 8)

    print("=== Speech Projector trainable parameters ===")
    for name, param in model.named_parameters():
        if "speech_projector" in name:
            print(name, "- requires_grad:", param.requires_grad)

    manifest = load_manifest(args.manifest_path)
    train_entries = [e for e in manifest if e["split"] == "train"]
    val_entries = [e for e in manifest if e["split"] == "val"]
    test_entries = [e for e in manifest if e["split"] == "test"]

    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}, Test: {len(test_entries)}")

    train_ds = SERDataset(train_entries, tokenizer, args.conv_mode, args.mel_size)
    val_ds = SERDataset(val_entries, tokenizer, args.conv_mode, args.mel_size)

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
        adam_beta2=0.95,
        warmup_ratio=0.01,
        lr_scheduler_type='cosine',
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        save_strategy='steps',
        save_steps=150,
        eval_strategy='steps',
        eval_steps=150,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=20,
        seed=3407,
        bf16=True,
        fp16=False,
        report_to="none",
        #deepspeed="/scratch/agarciam/tfm/models/Latxa-Omni-Emotion/omni_speech/train/ds_config.json",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks= [EarlyStoppingCallback(early_stopping_patience=3), EarlyEvalCallback()]
    )

    trainer.train()

    history = trainer.state.log_history

    steps = [x['step'] for x in history if 'loss' in x]
    loss = [x['loss'] for x in history if 'loss' in x]

    eval_steps = [x['step'] for x in history if 'eval_loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]

    plt.figure(figsize=(10, 6))
    loss_smooth = pd.Series(loss).rolling(window=20, min_periods=1).mean()
    plt.plot(steps, loss_smooth, label='Train Loss', color='red', alpha=0.6)
    plt.plot(eval_steps, eval_loss, label='Eval Loss', color='blue', marker='o')
    plt.title('Learning Curve - Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curve_loss.png")
    print("Graphic saved")

    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Latxa-3.1-8B-Omni")
    parser.add_argument("--model-base", type=str, default="Latxa-3.1-8B-Omni")
    parser.add_argument("--manifest-path", type=str, default="manifest.jsonl")
    parser.add_argument("--conv_mode", type=str, default="llama_3")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=False)
    parser.add_argument("--is_lora", action="store_true", default=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default='saves')
    args = parser.parse_args()
    fine_tuning(args)
    
'''dataset = SERDataset(train_entries[:5], tokenizer, "llama_3", 128)
example = dataset[0]
print(example.keys())
print("input_ids shape:", example["input_ids"].shape)
print("speech shape:", example["speech"].shape)
print("speech_lengths:", example["speech_lengths"])'''