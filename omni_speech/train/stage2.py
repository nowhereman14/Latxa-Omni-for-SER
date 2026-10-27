import argparse
import os
# import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #（代表仅使用第0，1号GPU）
import torch
from torch.utils.data import Dataset, DataLoader
import whisper
import numpy as np
# import ipdb  
import math
import json
from omni_speech.conversation import conv_templates
from omni_speech.model.builder import load_pretrained_model,create_model
from omni_speech.datasets.preprocess import tokenizer_speech_token
# from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
from tqdm import tqdm
import torch.optim as optim
# from memory_profiler import profile
# from transformers import DataCollatorForSeq2Seq
import os
from torch.nn.utils.rnn import pad_sequence

# Custom dataset class

def collate_fn(batch_data):
    for i in range(len(batch_data)):
        batch_data[i] = batch_data[i].values()
    input_ids,labels,speech_tensors, tgt_units,speech_lengths = zip(*batch_data)

    # input_idspad为llama的<|eot_id|>
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=128009)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    tgt_units = pad_sequence(tgt_units, batch_first=True, padding_value=-100)
    # input_ids = torch.stack(input_ids, dim=0)
    # labels = torch.stack(labels, dim=0)
    speech_tensors = torch.stack(speech_tensors, dim=0)
    speech_lengths = torch.stack(speech_lengths, dim=0)
    #转fp16
    
    ret=dict(input_ids=input_ids,labels=labels, speech=speech_tensors.bfloat16(), tgt_units = tgt_units, speech_lengths=speech_lengths)
    return ret

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, model_config, input_type, mel_size, data_root):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size
        self.data_root = data_root

    
    # def get_tgt_unit(self, file_path):
    #     unique_data_list = [] 
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         for line in file:
    #             line = line.strip()
    #             parts = line.split('<')
    #             result = [part for part in parts if part and '>' in part]
    #             # 移除元素末尾的 '>'
    #             result = [part.split('>')[0] for part in result]
    #             line_list = [int(item) for item in result]                
    #             #unique_data = [line_list[i] for i in range(len(line_list)) if i == 0 or line_list[i] != line_list[i-1]]
    #             unique_data_list.append(line_list)
    #     # return torch.tensor(unique_data_list)
    #     return unique_data_list

    def __getitem__(self, index):
        #tgt_unit = torch.tensor(self.tgt_unit[index])
        item = self.dataset[index]
        qs = "<speech>\nPlease directly answer the questions in the user's speech."
        re = item["answer"]
        # llm_gt = self.llm_gt[index]
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], re)
        prompt = conv.get_prompt()

        tgt_unit = np.load(os.path.join(args.data_root, item['answer_token']))
        tgt_unit = torch.tensor(tgt_unit)
        speech_file = item["question_audio"]

        speech = whisper.load_audio(os.path.join(self.data_root, speech_file))
        if self.input_type == "raw":
            speech = torch.from_numpy(speech)
            if self.model_config.speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
        elif self.input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)
        input_ids_ = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')
        input_ids = input_ids_.tolist()
        # 处理 input_ids 和 labels，仅训练answer部分的loss 
        split_markers = [128006, 78191, 128007, 271]
        last_marker_index = -1

        for i in range(len(input_ids) - len(split_markers) + 1):
            if input_ids[i:i + len(split_markers)] == split_markers:
                last_marker_index = i + len(split_markers)
                break
        if last_marker_index != -1:
            list1 = input_ids[:last_marker_index]
            list2 = input_ids[last_marker_index:]

        labels = len(list1) * [-100] + list2
        labels = torch.tensor(labels, device=input_ids_.device, dtype=input_ids_.dtype)
        ret=dict(input_ids=input_ids_,labels=labels, speech=speech, tgt_units=tgt_unit ,speech_lengths=torch.LongTensor([speech.shape[0]]))
        # ret=dict(input_ids=input_ids,labels=None, speech=speech, tgt_units=tgt_unit ,speech_lengths=torch.LongTensor([speech.shape[0]]))
        return ret
    def __len__(self):
        return len(self.dataset)

# DataLoader
def create_data_loader(dataset, tokenizer, model_config, input_type, mel_size, data_root, batch_size=1, num_workers=1):
    assert batch_size == 1, "batch_size must be 1"

    dataset = CustomDataset(dataset, tokenizer, model_config, input_type, mel_size, data_root)
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
    if 'WORLD_SIZE' in os.environ:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer, model, context_len = create_model(args.model_path, args.model_base, device=device, is_lora=args.is_lora, s2s=args.s2s)

    # train file
    # train_questions = json.load(open(os.path.expanduser(args.train_question_file), "r"))
    # train_questions = get_chunk(train_questions, args.num_chunks, args.chunk_idx) #chunk 1 chunk-idx 0 取list中的多少进行测试
    # with open(os.path.expanduser(args.train_answer_file), "r") as f:
    #     train_responses = f.readlines()
    #     for i in range(len(train_responses)):
    #         train_responses[i] = json.loads(train_responses[i])
    # data_loader = create_data_loader(questions,responses, tokenizer, model.config, args.input_type, args.mel_size)
    ds = load_from_disk(args.train_file)

    train_dl = create_data_loader(ds['train'], tokenizer, model.config, args.input_type, args.mel_size, args.data_root)
    eval_dl = create_data_loader(ds['test'], tokenizer, model.config, args.input_type, args.mel_size, args.data_root)

    # optimizer = optim.Adam(model.parameters(), lr=0.00001)
    # 学习率变大
    # optimizer = optim.Adam(model.speech_generator.parameters() , lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze only the generator
    for _, param in model.speech_generator.named_parameters():
        param.requires_grad = True


    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable parameters:", count_trainable_params(model))

    # 初始化Trainer
    model.train()
    training_args = TrainingArguments(
        output_dir=args.output_dir,                         # 输出路径，包括模型检查点、中间文件等
        overwrite_output_dir=True,                  # 是否覆写 output_dir
        do_train=True,                              # 是否做训练
        do_eval=False,                               # 是否做评估
        per_device_train_batch_size=args.train_batch_size,                
        gradient_accumulation_steps=args.gradient_accumulation_steps,    # 梯度累计步大小，省显存，但小模型没必要，用 1 收敛比较快
        per_device_eval_batch_size=args.eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps, 
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        # eval_steps=100,                            # 评估步骤间隔
        learning_rate=2e-4,                         # 学习率大小
        lr_scheduler_type='cosine',                 # 学习率调度策略，LLM 训练一般都用余弦
        bf16=True,        
        fp16=False,   
        half_precision_backend='cuda_amp',
        logging_steps=1,                           # 打印步骤间隔
        report_to='wandb',
        run_name=args.run_name,                             # 日志输出目标，不想用 wandb 可以设置为 None
        num_train_epochs=args.num_train_epochs,                         # 训练轮数，2 ~ 3 即可
        # save_steps=1000,                          # 检查点保存步骤间隔
        save_total_limit=2,                         # output_dir 内留存的检查点最大数目
        seed=3407,                                  # 随机种子
        max_grad_norm=1.0,
        save_strategy='epoch',
        # eval_strategy='epoch',
        logging_strategy='steps',
        # load_best_model_at_end=True,
        deepspeed="/home/asudupe/Latxa-Omni/omni_speech/train/ds_config_stage2.json",
    )
    tokenizer.pad_token = tokenizer.eos_token
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dl,
        eval_dataset=eval_dl,
        data_collator=collate_fn,
        # optimizers=(optimizer, None)
    )
    # with torch.no_grad:
    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser()
    #parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-path", type=str, default="Llama-3.1-8B-Omni")
    parser.add_argument("--model-base", type=str, default='Llama-3.1-8B-Omni')
    # parser.add_argument("--question-file", type=str, default="./omni_speech/infer/examples/question.json")
    parser.add_argument("--train-file", type=str, default="omni_speech/infer/gen_answer_data/question.json")
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
    parser.add_argument("--s2s", action="store_true", default=True)
    parser.add_argument("--is_lora",type=bool, default=False)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default='saves')
    parser.add_argument("--run_name", type=str)
    #parser.add_argument("--local-rank")
    args = parser.parse_args()
    train_model(args)
