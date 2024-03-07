from typing import Dict, Optional, Sequence
import json
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
)
import tqdm 
import copy
import logging 
import os
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

IGNORE_INDEX = -100
PAD_TOKEN = "[PAD]"
EOS_TOKEN = "</s>"
BOS_TOKEN = "<s>"
UNK_TOKEN = "<unk>"
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
        in_embeddings_avg = in_embeddings[ :-num_new_tokens].mean(dim=0, keepdim=True)
        out_embeddings_avg = out_embeddings[ :-num_new_tokens].mean(dim=0, keepdim=True)
        
        in_embeddings[-num_new_tokens: ] = in_embeddings_avg
        out_embeddings[-num_new_tokens: ] = out_embeddings_avg
        

def tokenize(string: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """ Tokenizer a list of strings"""
    
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
) -> Dict:
    """Preprocess the data by tokenizing."""
    
    examples = [s + t for s, t in zip(sources, targets)]
    if not os.path.exists("alpaca_tokenized.pt"):
        tokenized = [tokenize(strings, tokenizer) for strings in tqdm.tqdm((examples, sources), desc="Tokenizing")]
        torch.save(tokenized, "alpaca_tokenized.pt")
    else:
        tokenized = torch.load("alpaca_tokenized.pt")
    
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
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
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
    

class SFTDataCollator():
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
        self.tp = 0.
        
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        logits, labels = logits.argmax(dim=-1).view(-1).cpu(), labels.view(-1).cpu()
        tp = (logits == labels).sum()
        self.count += len(logits)
        self.tp += tp
        return tp / len(logits)
    
    def compute(self):
        return self.tp / self.count