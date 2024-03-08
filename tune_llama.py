# Finetunes LLaMA-7B on Alpaca dataset. Supports DDP via torchrun.
# Disclaimer: Referenced https://github.com/tatsu-lab/stanford_alpaca/tree/main/train.py
# Ex: torchrun --nproc_per_node 4 master_port 25555 tune_llama.py --dp_size 2 --tp_size 2
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    LlamaConfig,
    LlamaTokenizer,
    PreTrainedTokenizer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

from src.data_utils import *
from src.modeling_llama import LlamaForCausalLM, tp_modules


def manual_reduction(
    model: nn.Module,
    group: dist.ProcessGroupNCCL = None,
    is_tp: bool = False,
    average: bool = True,
):
    """Manually reduce the gradient within a group. I suspect DDP's backward is buggy when there are multiple process groups."""
    for name, param in model.named_parameters():
        # TP layers already have grads synchronized, just all-reduce non-parallel ones
        if is_tp and any([tp_name in name for tp_name in tp_modules]):
            continue

        if param.requires_grad and param.grad is not None:
            # All-reduce gradients manually within the specified group
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
            if average:
                param.grad.data /= dist.get_world_size(group)


def init_dist(dp_size: int = -1) -> List[dist.ProcessGroupNCCL]:
    dp_size = int(os.environ["WORLD_SIZE"]) if dp_size == -1 else dp_size
    rank = int(os.environ["LOCAL_RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    tp_size = world_size // dp_size

    dist.init_process_group(world_size=world_size, rank=rank, init_method="env://", backend="nccl")
    torch.cuda.set_device(local_rank)

    # Set up Tensor Parallel within each Data Parallel group
    tp_groups = dp_groups = None
    if dp_size > 1:
        tp_groups = [
            dist.new_group(ranks=range(group_id * tp_size, (group_id + 1) * tp_size)) for group_id in range(dp_size)
        ]

    # Set up Data Parallel groups across Tensor Parallel groups
    # e.g. 4 GPUs, 2 TP groups, 2 DP groups, rank = {0, 1, 2, 3}
    # TP groups: {0, 1}, {2, 3}
    # DP groups: {0, 2}, {1, 3}
    if tp_size > 1:
        dp_groups = [dist.new_group(ranks=range(i, world_size, tp_size)) for i in range(tp_size)]

    # Assign each rank a TP and DP group
    rank = int(os.environ["LOCAL_RANK"])
    tp_group = tp_groups[rank // tp_size] if tp_groups is not None else None
    dp_group = dp_groups[rank % dp_size] if dp_groups is not None else None

    return tp_group, dp_group


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/home/zhongyuting/model/Llama-2-7b-hf",
        metadata={"help": "Local path to the model or path in huggingface directory"},
    )
    tp_size: int = field(default=1, metadata={"help": "Size of tensor parallel groups."})
    dp_size: int = field(default=1, metadata={"help": "Size of data parallel groups."})
    debug_mode: bool = field(default=False, metadata={"help": "Debug by printing layer info etc."})
    manual_dp: bool = field(
        default=True,
        metadata={"help": "Manually reduce gradients instead of using torch.DDP."},
    )


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
    weight_decay: float = 0.0
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


def get_data(tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> Dict:
    train_dataset = SFTDataset(data_args.data_path, tokenizer)
    collator = SFTDataCollator(tokenizer)
    return train_dataset, collator


def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(x.reshape(-1, x.shape[-1]), y.reshape(-1))


def main(tp_group: dist.ProcessGroupNCCL, dp_group: dist.ProcessGroupNCCL):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tp_size = model_args.tp_size
    dp_size = model_args.dp_size

    world_size = int(os.environ["WORLD_SIZE"])
    assert (
        world_size == tp_size * dp_size
    ), f"Num DP groups * TP size must equal num devices, but we have {os.environ['WORLD_SIZE']} != {tp_size * dp_size}"

    # Setup NCCL Comm groups
    rank = int(os.environ["LOCAL_RANK"])

    # Load model and set up parallelism
    config = LlamaConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    config.tp_size = tp_size
    config.debug_mode = model_args.debug_mode

    # config.comm_group = tp_group
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
    )

    # Enable tensor parallel
    if config.tp_size > 1:
        for module in model.modules():
            if hasattr(module, "set_tp"):
                module.set_tp(comm_group=tp_group)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Save memory by freezing params
    # Dumb pytorch won't allow partially freezing in DDP
    if not dp_size > 1:
        n_freeze = 15
        for param in model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = True
        for param in model.model.layers[n_freeze:].parameters():
            param.requires_grad = True
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
    # Check for bug in DDP
    if rank == 0:
        for name, param in model.named_parameters():
            if param.device == torch.device("cpu"):
                print(name, "is not on GPU!!")

    # NOTE: test manually reduce gradients
    if dp_size > 1 and not model_args.manual_dp:
        # Must NOT set device with MP + DP, as instructed by torch docs (https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
        if tp_size > 1:
            model = DDP(model)

    # Set up data, optimizer, scheduler
    train_dataset, data_collator = get_data(tokenizer=tokenizer, data_args=data_args)
    sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=dp_size,
            rank=rank // tp_size,
        )
        if dp_size > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_training_batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        sampler=sampler,
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.lr,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=training_args.weight_decay,
    )
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

    # Set mixed precision
    if training_args.bf16:
        dtype = torch.bfloat16
    elif training_args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    if rank == 0:
        pbar = tqdm.tqdm(total=total_steps)
        print(
            f"Actual batch size: {training_args.per_device_training_batch_size * dp_size * training_args.gradient_accumulation_steps}"
        )

    for epoch in range(training_args.epochs):
        if sampler:
            train_loader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            batch = to_gpu(batch)
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model(**batch)
                loss = loss_fn(out.logits, batch["labels"]) / training_args.grad_accumulation_steps
                loss.backward()  # TODO: DDP + TP could hang during backward??

                # All-reduce grads of non-parallel modules
                if tp_size > 1:
                    manual_reduction(model, tp_group, is_tp=True)

                # All-reduce across DP groups
                if model_args.manual_dp and dp_size > 1:
                    manual_reduction(model, dp_group)

            if step % training_args.grad_accumulation_steps == 0:
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                train_step += 1

                # Update tqdm postfix with training metrics
                if rank == 0:
                    pbar.set_postfix(
                        {
                            "Step": train_step,
                            "Loss": loss.item() * training_args.gradient_accumulation_steps,
                            "Accuracy": acc.update(out.logits, batch["labels"]),
                            "lr": scheduler.get_last_lr()[0],
                        },
                        refresh=True,
                    )
            if rank == 0:
                pbar.update(1)


if __name__ == "__main__":
    import sys

    dp_size = 1  # default
    if "--dp_size" in sys.argv:
        dp_index = sys.argv.index("--dp_size")
        dp_size = int(sys.argv[dp_index + 1])

    tp_group, dp_group = init_dist(dp_size)
    main(tp_group, dp_group)
