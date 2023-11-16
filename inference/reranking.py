import os
import csv
import json
import torch
import argparse
import transformers
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from transformers import (
    LlamaForSequenceClassification,
    LlamaTokenizerFast,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
parser = argparse.ArgumentParser()

parser.add_argument("--device_id", type = int, default = 3)
parser.add_argument("--lora_r", type = int, default = 32)
parser.add_argument("--input_data", type = str, default = 'alpaca7b.json')
parser.add_argument("--save_generations", type = str, default = None)
parser.add_argument("--reward_model_name", type = str, default = None)
parser.add_argument("--reward_model_path", type = str, default = None)
parser.add_argument("--alpaca_model_path", type = str, default = "alpaca7b")

args = parser.parse_args()

current_device = args.device_id

tokenizer  = LlamaTokenizerFast.from_pretrained(args.alpaca_model_path)
tokenizer.pad_token_id = (0)
tokenizer.padding_side = "left" 
model  = LlamaForSequenceClassification.from_pretrained(args.alpaca_model_path, num_labels = 1, load_in_8bit=True, torch_dtype=torch.float16, device_map={"": current_device})
model = PeftModel.from_pretrained(model, args.reward_model_path, device_map={"": current_device})
print('Model Loaded')
model.eval()

PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

with open(args.input_data, 'r') as f:
    data = json.load(f)

with torch.no_grad():
    results = []
    for example in tqdm(data):
        response = {}
        response['dataset']   = example['dataset']
        response['generator'] = args.reward_model_name
        instruction = example['instruction']
        response['instruction'] = instruction
        outputs = example['outputs']
        sentences = []
        for output in outputs:
            sentence = PROMPT_DICT['prompt_no_input'].format(instruction = instruction) + output
            sentences.append(sentence)
        rewards = []
        for i in range(0, len(sentences), 8):
            input = tokenizer(sentences[i:i+8], truncation = True, padding = 'max_length', max_length = 512, return_tensors = "pt")
            input_ids = input['input_ids'].to(current_device)
            attention_mask = input['attention_mask'].to(current_device)
            temp   = model(input_ids = input_ids, attention_mask = attention_mask)['logits'].squeeze()
            rewards.append(temp)
        rewards = torch.cat(rewards)
        max_index = rewards.argmax().item()
        response['output'] = outputs[max_index]
        results.append(response)
        with open(args.save_generations, 'w') as f:
            json.dump(results, f)

if __name__ == "main":
    main()