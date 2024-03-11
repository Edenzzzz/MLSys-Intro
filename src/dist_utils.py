import copy
import json
import logging
import os
from typing import Dict, List, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer


def init_dist(dp_size: int = -1) -> List[dist.ProcessGroupNCCL]:
    """
    Initialize distributed training, with optional Data Parallel and Tensor Parallel sizes.
    Arguments:
        dp_size: Number of data parallel groups. By default will not use DP  and will use
            all devices for TP.
    """

    dp_size = int(os.environ["WORLD_SIZE"]) if dp_size == -1 else dp_size
    rank = int(os.environ["LOCAL_RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    tp_size = world_size // dp_size

    dist.init_process_group(world_size=world_size, rank=rank, init_method="env://", backend="nccl")
    torch.cuda.set_device(local_rank)

    # Set up Tensor Parallel groups
    tp_groups = dp_groups = None
    if dp_size > 1:
        tp_groups = [
            dist.new_group(ranks=range(group_id * tp_size, (group_id + 1) * tp_size)) for group_id in range(dp_size)
        ]

    # Set up Data Parallel across Tensor Parallel groups
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


def to_gpu(tensor_dict):
    device = dist.get_rank() if dist.is_initialized() else "cuda"
    return {k: v.to(device, non_blocking=True) for k, v in tensor_dict.items()}


def print_once(*message: str):
    """Ensure printing only once in a distributed setting"""
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print(*message)


def check_on_gpu(model: nn.Module):
    for name, param in model.named_parameters():
        if param.device == torch.device("cpu"):
            print(name, "is not on GPU!!")


####################### Llama utils #######################
IGNORE_INDEX = -100
PAD_TOKEN = "[PAD]"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def resize_tokenizer_embedding(
    new_tokens: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """
    Resize emebdding and tokenizer.
    """
    num_new_tokens = tokenizer.add_special_tokens(new_tokens)
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

        # Assign new embeddings the average value
        in_embeddings = model.get_input_embeddings().weight.data
        out_embeddings = model.get_output_embeddings().weight.data
        in_embeddings_avg = in_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        out_embeddings_avg = out_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        in_embeddings[-num_new_tokens:] = in_embeddings_avg
        out_embeddings[-num_new_tokens:] = out_embeddings_avg


def tokenize(string: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenizer a list of strings"""

    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in string
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
    path: str = "alpaca_tokenized.pt",
) -> Dict:
    """Preprocess the data by tokenizing."""

    examples = [s + t for s, t in zip(sources, targets)]

    def _tokenize_all():
        if not os.path.exists(path):
            tokenized = [tokenize(strings, tokenizer) for strings in tqdm.tqdm((examples, sources), desc="Tokenizing")]
            torch.save(tokenized, path)
        else:
            tokenized = torch.load(path)
        return tokenized

    # Only preprocess on rank 0 and put others waiting on barrier
    if dist.is_initialized():
        if dist.get_rank() == 0:
            tokenized = _tokenize_all()
        dist.barrier()
        if dist.get_rank() != 0:
            tokenized = torch.load(path)

    examples_tokenized, sources_tokenized = tokenized
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning"""

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer):
        super(SFTDataset, self).__init__()

        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))
        logging.warning("Formatting inputs...")

        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        sources = [
            (
                prompt_input.format_map(example)
                if example.get("input", "") != ""
                else prompt_no_input.format_map(example)
            )
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class SFTDataCollator:
    """Collate batch for SFT"""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class Accuracy:
    "A simple Accuracy function compatible with HF models"

    def __init__(self):
        self.count = 0
        self.tp = 0.0

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        logits, labels = logits.argmax(dim=-1).view(-1).cpu(), labels.view(-1).cpu()
        tp = (logits == labels).sum()
        self.count += len(logits)
        self.tp += tp
        return tp / len(logits)

    def compute(self):
        return self.tp / self.count


def generate(model, prompt, tokenizer, max_new_tokens=100):
    with torch.inference_mode():
        tokenized_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        output = model.generate(tokenized_prompt, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0][len(tokenized_prompt[0]) :], skip_special_tokens=True)
