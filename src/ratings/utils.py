from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import csv
import torch
import pickle
import pandas as pd
import numpy as np
from torch import nn
from transformers import Trainer
from torch.utils.data import Dataset

from datasets import load_metric
from collections import defaultdict
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def custom_collate(batch):
    batched_item = defaultdict(list)
    for item in batch:
        for key in item:
            batched_item[key].append(item[key])
    batched_item['input_ids'] = torch.stack(batched_item['input_ids'], dim = 0)
    batched_item['attention_mask'] = torch.stack(batched_item['attention_mask'], dim = 0)
    return batched_item

class RewardDataset(Dataset):

    def __init__(self, data_file, tokenizer):

        self.data_file = data_file
        df = pd.read_csv(self.data_file, names = ['instruction', 'input', 'output', 'score'])
        self.instructions = df['instruction'].tolist()
        self._inputs = df['input'].tolist()
        self.outputs = df['output'].tolist()
        self.model_inputs = []
        for j in range(len(self.instructions)):
            if pd.isnull(self._inputs[j]):
                model_input = PROMPT_DICT['prompt_no_input'].format(instruction = str(self.instructions[j])) + str(self.outputs[j])
            else:
                model_input = PROMPT_DICT['prompt_input'].format(instruction = str(self.instructions[j]), input = str(self._inputs[j])) + str(self.outputs[j])
            self.model_inputs.append(model_input)
        self.tokenizer = tokenizer
        self.inputs = self.tokenizer(self.model_inputs, truncation = True, padding = 'max_length', max_length = 512, return_tensors = "pt")
        self.scores = df['score'].tolist()
    
    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, index):
        item = {}
        item['input_ids'] = self.inputs['input_ids'][index]
        item['attention_mask'] = self.inputs['attention_mask'][index]
        item['target'] = (self.scores[index] - 1) / 6
        item['instruction'] = self.instructions[index]
        item['_input'] = str(self._inputs[index])
        item['output'] = self.outputs[index]
        return item

class RewardTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids'].to(self.args.device)
        attention_mask = inputs['attention_mask'].to(self.args.device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)['logits'].squeeze()
        target = torch.tensor(inputs['target']).to(self.args.device)
        sigm = torch.nn.Sigmoid()
        loss = (target - sigm(logits)).square().mean()
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        with torch.no_grad():
            instructions = inputs['instruction']
            _inputs = inputs['_input']
            outputs = inputs['output']
            input_ids = inputs['input_ids'].to(self.args.device)
            attention_mask = inputs['attention_mask'].to(self.args.device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)['logits'].squeeze()
            target = torch.tensor(inputs['target']).to(self.args.device)
            sigm = torch.nn.Sigmoid()
            pred = sigm(logits)
            loss = (target - pred).square().mean()
            if prediction_loss_only:
                return (loss, None, None)
            else:
                return (loss, logits, target)