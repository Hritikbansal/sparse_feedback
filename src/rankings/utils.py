from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import csv
import json
import torch
import pickle
import pandas as pd
import numpy as np
from torch import nn
from transformers import Trainer
from torch.utils.data import Dataset

from datasets import load_metric
from collections import defaultdict


class RewardDataset(Dataset):

    def __init__(self, data_file, tokenizer):

        self.data_file = data_file
        with open(self.data_file, 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.keys = list(filter(lambda x: len(self.data[x]['sentences']) <= 10, self.keys))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        item = {}
        instruction = self.keys[index]
        item['instruction'] = instruction
        sentence_pairs = self.data[instruction]['sentences']
        item['sentences'] = []
        item['sentences_tokenized'] = []
        for sentence_pair in sentence_pairs:
            item['sentences'].append(sentence_pair)
            sentence_pair = [instruction + sp for sp in sentence_pair]
            input = self.tokenizer(sentence_pair, truncation = True, padding = 'max_length', max_length = 512, return_tensors = "pt") ## {'input_ids': , 'attention_mask': }
            item['sentences_tokenized'].append(input)  
        return item

def custom_collate(batch):
    batched_item = defaultdict(list)
    for item in batch:
        for j in range(len(item['sentences'])):
            batched_item['instruction'].append(item['instruction'])
            batched_item['sentences'].append(item['sentences'][j])
            batched_item['input_ids'].append(item['sentences_tokenized'][j]['input_ids'])
            batched_item['attention_mask'].append(item['sentences_tokenized'][j]['attention_mask'])
    batched_item['input_ids'] = torch.cat(batched_item['input_ids'], dim = 0)
    batched_item['attention_mask'] = torch.cat(batched_item['attention_mask'], dim = 0)
    return batched_item

class RewardTrainer(Trainer):

    # Define how to compute the reward loss.
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids'].to(self.args.device)
        attention_mask = inputs['attention_mask'].to(self.args.device)
        rewards = model(input_ids=input_ids, attention_mask=attention_mask)['logits'].squeeze()
        rewards_pos, rewards_neg = rewards[::2], rewards[1::2]
        sigm = torch.nn.Sigmoid()
        loss = -torch.log(sigm(rewards_pos - rewards_neg)).mean()
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        with torch.no_grad():
            input_ids = inputs['input_ids'].to(self.args.device)
            attention_mask = inputs['attention_mask'].to(self.args.device)
            rewards = model(input_ids=input_ids, attention_mask=attention_mask)['logits'].squeeze()
            rewards_pos, rewards_neg = rewards[::2], rewards[1::2]
            instructions = inputs['instruction']
            sentences = inputs['sentences']
            # if self.save_file:
            #     for j in range(len(rewards_pos)):
            #         with open(self.save_file, 'a') as f:
            #             csvwriter = csv.writer(f)
            #             csvwriter.writerow([instructions[j], sentences[j][0], str(rewards_pos[j].item())])
            #             csvwriter.writerow([instructions[j], sentences[j][1], str(rewards_neg[j].item())])
            sigm = torch.nn.Sigmoid()
            loss = -torch.log(sigm(rewards_pos - rewards_neg)).mean()
            return (loss, None, None)