# Finetunes LLaMA-7B on Alpaca dataset. Supports DDP via torchrun.
# Disclaimer: Referenced https://github.com/tatsu-lab/stanford_alpaca/tree/main/train.py
# Ex: torchrun --nproc_per_node 4 master_port 25555 tune_llama.py --dp_size 2 --tp_size 2
import copy
import torch 
from transformers import (
    TrainingArguments,
    LlamaTokenizer,
    PreTrainedTokenizer,
    get_cosine_schedule_with_warmup,
    LlamaConfig
)
from src.modeling_llama import LlamaForCausalLM
import transformers
from torch.utils.data import DataLoader   
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import os
import torch.nn.functional as F
from src.data_utils import *
import tqdm

def init_dist(dp_size: int = -1) -> List[dist.ProcessGroupNCCL]:
    dp_size = int(os.environ["WORLD_SIZE"]) if dp_size == -1 else dp_size
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE']) 
    dist.init_process_group(world_size=world_size, rank=rank,
                        init_method="env://", backend="nccl")
    
    torch.cuda.set_device(local_rank)
    
    # Set up Tensor Parallel within each Data Parallel group
    tp_size = world_size // dp_size
    if dp_size > 1:
        dp_groups = [dist.new_group(ranks=range(group_id * tp_size, (group_id + 1) * tp_size)) for group_id in range(dp_size)]
        return dp_groups

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/zhongyuting/model/Llama-2-7b-hf", metadata={"help": "Local path to the model or path in huggingface directory"})
    tp_size: int = field(default=1, metadata={"help": "Size of tensor parallel groups."})
    dp_size: int = field(default=1, metadata={"help": "Size of data parallel groups."})
    debug_mode: bool = field(default=False, metadata={"help": "Debug by printing layer info etc."})
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
    grad_accumulation_steps: int = 4
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


def main(dp_groups: List[dist.ProcessGroupNCCL] = None):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    world_size = int(os.environ["WORLD_SIZE"])
    assert  world_size == model_args.tp_size * model_args.dp_size,\
        f"Num DP groups * TP size must equal num devices, but we have {os.environ['WORLD_SIZE']} != {model_args.tp_size * model_args.dp_size}"
    
    # Setup NCCL Comm groups
    rank = int(os.environ["LOCAL_RANK"])
    this_group = dp_groups[rank // model_args.tp_size] if dp_groups is not None else None

    # Load model and set up parallelism     
    config = LlamaConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    config.tp_size = model_args.tp_size
    # config.comm_group = this_group
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
    )
                    
    # Enable tensor parallel
    if config.tp_size > 1:
        for module in model.modules():
            if hasattr(module, "set_tp"):
                module.set_tp(comm_group=this_group)
                
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Save memory by freezing params
    # Dumb pytorch won't allow partially freezing in DDP
    if not model_args.dp_size > 1: 
        n_freeze = 15
        for param in model.parameters(): param.requires_grad = False
        for param in model.lm_head.parameters(): param.requires_grad = True
        for param in model.model.layers[n_freeze: ].parameters(): param.requires_grad = True
    if rank == 0:
        print("Config: ", model.config)
    
    
     # Load vocab and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        # pad_token = PAD_TOKEN,
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
    
    model = model.cuda()
    if rank == 0:
        for name, param in model.named_parameters():
            if param.device == torch.device("cpu"):
                print(name, "is not on GPU!!")

    if model_args.dp_size > 1:
        device = "cuda:" + os.environ['LOCAL_RANK']
        model = DDP(model, device_ids=[device], output_device=device)
        

    # Set up data, optimizer, scheduler
    train_dataset, data_collator = get_data(tokenizer=tokenizer, data_args=data_args)
    sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(this_group), rank=dist.get_rank(this_group)) if model_args.dp_size > 1 else None
    train_loader = DataLoader(train_dataset,
                              batch_size=training_args.per_device_training_batch_size,
                              collate_fn=data_collator,
                              pin_memory=True,
                              sampler=sampler
                            ) 
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
    if rank == 0:
        pbar = tqdm.tqdm(total=total_steps)
    for epoch in range(training_args.epochs):
        if sampler:
            train_loader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            batch = to_gpu(batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = loss_fn(out.logits, batch["labels"]) / training_args.grad_accumulation_steps  
                loss.backward()
                
            if step % training_args.grad_accumulation_steps == 0:
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                train_step += 1
                
                # Update tqdm postfix with training metrics
                if rank == 0:
                    pbar.set_postfix({
                        "Step": train_step,
                        "Loss": loss.item() * training_args.gradient_accumulation_steps,
                        "Accuracy": acc.update(out.logits, batch["labels"]),
                        "lr": scheduler.get_last_lr()[0]
                    }, refresh=True)
            if rank == 0:   
                pbar.update(1)


if __name__ == "__main__":
    import sys
    dp_size = 1 # default
    if "--dp_size" in sys.argv:
        dp_index = sys.argv.index("--dp_size")
        dp_size = int(sys.argv[dp_index + 1])
        
    dp_groups = init_dist(dp_size)
    main(dp_groups)    