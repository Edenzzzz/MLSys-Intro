# Finetunes LLaMA-7B on Alpaca dataset. Supports DDP via torchrun.
import copy
import torch 
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_cosine_schedule_with_warmup,
)
import transformers
from transformers import GenerationConfig
from torch.utils.data import DataLoader   
import json
from typing import Dict, Optional, Sequence
from dataclasses import dataclass, field
import os
import torch.nn.functional as F
from data_utils import *
import tqdm


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/zhongyuting/model/Llama-2-7b-hf", metadata={"help": "Local path to the model or path in huggingface directory"})
    tp_num_devices: int = field(default=1, metadata={"help": "Tensor parallel device count"})
    
@dataclass
class DataArguments:
    data_path: str = field(default="alpaca_data.json", metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded/truncated)."},
    )

    # Hyperparams
    lr: float = 2e-5
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type = "cosine"
    epochs: int = 3
    per_device_training_batch_size: int = 4
    per_device_eval_batch_size: int = 16
    
    # Memory optimizations
    gradient_checkpointing: bool = True
    grad_accumulation_steps: int = 8
    bf16: bool = False
    fp16: bool = False    
    
    # Logging
    evaluation_strategy: str = "no"
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 1
    logging_steps: int = 100
    output_dir: str = "checkpoints"


def to_gpu(tensor_dict):
    return {k: v.to('cuda', non_blocking=True) for k, v in tensor_dict.items()}

def get_data(tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> Dict:
    train_dataset = SFTDataset(data_args.data_path, tokenizer)
    collator = SFTDataCollator(tokenizer)
    return train_dataset, collator


def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(x.reshape(-1, x.shape[-1]), y.reshape(-1))


def generate(model, prompt, tokenizer, max_new_tokens=100):
    with torch.inference_mode():
        tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        output = model.generate(tokenized_prompt, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        pretraining_tp=model_args.tp_num_devices,
    ).to("cuda")
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    # Save memory by freezing params
    n_freeze = 20
    for param in model.parameters(): param.requires_grad = False
    for param in model.lm_head.parameters(): param.requires_grad = True
    for param in model.model.layers[n_freeze: ].parameters(): param.requires_grad = True
    
    print("Config: ", model.config)
    
    # Load vocab and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # Llama has no padding token by default, so add it
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = PAD_TOKEN

    resize_tokenizer_embedding(
        new_tokens=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # Set up data, optimizer, scheduler
    train_dataset, data_collator = get_data(tokenizer=tokenizer, data_args=data_args)
    train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_training_batch_size, collate_fn=data_collator)
    optim = torch.optim.AdamW(model.parameters(), lr=training_args.lr, betas=(0.9, 0.99), eps=1e-6, weight_decay=training_args.weight_decay)
    total_steps = training_args.epochs * len(train_loader) // training_args.grad_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * training_args.warmup_ratio),
        num_training_steps=total_steps,
    )

    
    # Training 
    acc = Accuracy()
    model.train()
    train_step = 0
    pbar = tqdm.tqdm(total=total_steps)
    for epoch in range(training_args.epochs):
        for step, batch in enumerate(train_loader):
            batch = to_gpu(batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = loss_fn(out.logits, batch["labels"]) / training_args.grad_accumulation_steps  # you could use out.loss and not shift the dataset  
                loss.backward()
                
            if step % training_args.grad_accumulation_steps == 0:
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                train_step += 1
                
                # Update tqdm postfix with training metrics
                pbar.set_postfix({
                    "Step": train_step,
                    "Loss": loss.item() * training_args.gradient_accumulation_steps,
                    "Accuracy": acc.update(out.logits, batch["labels"]),
                    "lr": scheduler.get_last_lr()[0]
                }, refresh=True)
            
            pbar.update(1)


if __name__ == "__main__":
    main()    