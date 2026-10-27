
import os
os.chdir("/home/asudupe/Latxa-Omni")

import sys

root = '/home/asudupe/Latxa-Omni'
sys.path.append(str(root))
# print(root)

from omni_speech.model.builder import load_pretrained_model,create_model
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import whisper
from omni_speech.conversation import conv_templates
# import ipdb  
import math
import json
# from tqdm import tqdm
from omni_speech.datasets.preprocess import tokenizer_speech_token
# from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
from tqdm import tqdm
# import torch.optim as optim
import torch.optim as optim
# from transformers import DataCollatorForSeq2Seq
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer
from omni_speech.metrics import compute_metrics
from torchaudio.transforms import Resample
from datasets import load_from_disk, load_dataset, Audio
from scipy.signal import resample_poly
import io
import soundfile as sf
import numpy as np



# Custom dataset class

def collate_fn(batch):
    for i in range(len(batch)):
        batch[i]= batch[i].values()
        
    input_ids,labels,speech_tensors,speech_lengths = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=128009)
    labels = pad_sequence(labels, batch_first=True, padding_value=128009)

    speech_tensors = torch.stack(speech_tensors, dim=0)
    speech_lengths = torch.stack(speech_lengths, dim=0)
    return {"input_ids":input_ids,"labels":labels, "speech":speech_tensors, "speech_lengths":speech_lengths}

class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer, model_config, input_type, mel_size, data_root):
        self.questions = questions
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size
        self.data_root = data_root

    def __getitem__(self, index):
        item = self.questions[index]
        # speech_file = item["speech"]
        # qs = item["conversations"][0]["value"]
        # re = item["conversations"][1]["value"]
        # audio_bytes = item["question_audio"]["bytes"]
        # audio, sr = sf.read(io.BytesIO(audio_bytes))
        # audio = resample_poly(audio, 16000, sr)
        # speech = torch.tensor(audio)
        # speech = speech.double()
        # print(speech)

        # audio = torch.tensor(item['question_audio'])
        # speech = Resample(orig_freq=22050, new_freq=16000)(audio)
        # speech = Resample(orig_freq=sr, new_freq=16000)(speech)

        speech_file = item['question_audio']
        speech = whisper.load_audio(os.path.join(self.data_root, speech_file))
        qs = "<speech>\nPlease directly answer the questions in the user's speech."
        re = item["answer"]

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], re)
        prompt = conv.get_prompt()

        # speech = whisper.load_audio(speech_file)
        if self.input_type == "raw":
            speech = torch.from_numpy(speech)
            if self.model_config.speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
        elif self.input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)
        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')
        ret=dict(input_ids=input_ids,labels=input_ids, speech=speech.to(torch.bfloat16), speech_lengths=torch.LongTensor([speech.shape[0]]))
        return ret
    def __len__(self):
        return len(self.questions)
    
# DataLoader
def create_data_loader(questions, tokenizer, model_config, input_type, mel_size, data_root, batch_size=2, num_workers=1):
    # assert batch_size == 1, "batch_size must be 1"
    
    dataset = CustomDataset(questions, tokenizer, model_config, input_type, mel_size, data_root)
    #data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return dataset


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def train_model(args):
    # 设置每张卡的device
     
    
    if 'WORLD_SIZE' in os.environ:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model_path = os.path.expanduser(args.model_path)
    # print(model_path)
    tokenizer, model, context_len = create_model(args.model_path, args.model_base, is_lora=args.is_lora, s2s=args.s2s, device=device)  

    # print('tokenizer----------------------------------')
    # print(tokenizer)
    # print('model----------------------------------')
    # print(model)

    # if tokenizer==False:
    #     print(tokenizer)
    #     if args.model_base==None:
    #         print(f'You must specify model_base. {args.model_base} not right')
    #         return 0
    #     print(args.model_base)
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_base, padding_side="right",padding=True, use_fast=True)
    
    #     print('tokenizer----------------------------------')
    #     print(tokenizer)

    # questions = json.load(open(os.path.expanduser(args.train_file), "r"))
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx) #chunk 1 chunk-idx 0 取list中的多少进行测试
    # train_dl = create_data_loader(questions, tokenizer, model.config, args.input_type, args.mel_size)

    # validations = json.load(open(os.path.expanduser(args.valid_file), "r"))
    # validations = get_chunk(validations, args.num_chunks, args.chunk_idx )
    # eval_dl = create_data_loader(validations, tokenizer, model.config, args.input_type, args.mel_size)

    ds = load_from_disk(args.train_file)
    # questions = load_dataset(args.train_file)
    # ds = ds.cast_column("question_audio", Audio(decode=False))

    # def decode_audio(example):
    #     audio_bytes = example["audio"]["bytes"]
    #     audio, sr = sf.read(io.BytesIO(audio_bytes))
    #     if audio.ndim > 1:
    #         audio = np.mean(audio, axis=1)

    #     # Resample from 24kHz to 16kHz
    #     if sr != 16000:
    #         audio = resample_poly(audio, 16000, sr)
    #         sr = 16000
    #     example["question_audio"] = audio
    #     example["sampling_rate"] = sr
    #     return example

    # ds = ds.map(decode_audio)

    train_dl = create_data_loader(ds['train'], tokenizer, model.config, args.input_type, args.mel_size, args.data_root)
    eval_dl = create_data_loader(ds['test'], tokenizer, model.config, args.input_type, args.mel_size, args.data_root)

    # from transformers import Trainer, TrainingArguments
    # 初始化Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,                 # 输出路径，包括模型检查点、中间文件等
        overwrite_output_dir=True,                  # 是否覆写 output_dir
        do_train=True,                              # 是否做训练
        do_eval=False,                               # 是否做评估
        per_device_train_batch_size=args.train_batch_size,                
        gradient_accumulation_steps=args.gradient_accumulation_steps,    # 梯度累计步大小，省显存，但小模型没必要，用 1 收敛比较快
        # per_device_eval_batch_size=args.eval_batch_size,
        # eval_accumulation_steps=args.eval_accumulation_steps,                  
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        learning_rate=2e-5,
        weight_decay=0.01,
        adam_beta2=0.95,
        warmup_ratio=0.01,
        lr_scheduler_type='cosine',                 # 学习率调度策略，LLM 训练一般都用余弦
        report_to="wandb",                          # 日志输出目标，不想用 wandb 可以设置为"none"
        run_name=args.run_name,    
        num_train_epochs=args.num_train_epochs,     
        # eval_strategy='epoch',
        logging_steps=1,                           # Print step interval
        save_strategy='epoch',
        # metric_for_best_model='accuracy',
        # greater_is_better=True,
        save_total_limit=2,                         # output_dir 内留存的检查点最大数目
        seed=3407,                                  # 随机种子
        bf16=True,                            # 是否开启混合精度训练 (V100: False)   
        fp16 = False,
        # attn_implementation = "sdpa",
        deepspeed="/home/asudupe/Latxa-Omni/omni_speech/train/ds_config.json",
    )
    tokenizer.pad_token = tokenizer.eos_token
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dl,
        eval_dataset=eval_dl,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    # try:
    trainer.train()
    # except:
    #     print('OOM')
    #     print(torch.cuda.memory_summary())
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--train-file", type=str)
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--valid-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_type", type=str, default="mel")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=False)
    parser.add_argument("--is_lora", action="store_true", default=False)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default='saves')
    parser.add_argument("--run_name", type=str)
    args = parser.parse_args()
    train_model(args)