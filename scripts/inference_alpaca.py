import csv
import json
import torch
import argparse
import transformers
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--device_id", type = int, default = 3)
parser.add_argument("--input", type = str, default = 'data.json')
parser.add_argument("--output", type = str, default = 'alpaca_output.json')
parser.add_argument("--model_path", type = str, default = "alpaca7b")

args = parser.parse_args()

model_path = args.model_path

model = transformers.AutoModelForCausalLM.from_pretrained(model_path, 
                                                            device_map = {"": torch.device(f"cuda:{args.device_id}")},
                                                            torch_dtype = torch.float16,
                                                            low_cpu_mem_usage=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)


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

def main():

    with open(args.input, 'r') as f:
        data = json.load(f)
    
    result = []
    for example in tqdm(data):
        instruction = example['instruction']
        input = "" if not 'input' in example else example["input"]
        input_text = PROMPT_DICT['prompt_input'].format(instruction = instruction, input = input) if 'input' in example else PROMPT_DICT['prompt_no_input'].format(instruction = instruction)
        inputs = tokenizer(input_text, return_tensors="pt")
        out = model.generate(inputs=inputs.input_ids.to(f"cuda:{args.device_id}"), max_new_tokens = 128, num_return_sequences=5, do_sample=True)
        output_texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        outs = []
        for output_text in output_texts:
            output_text = output_text[len(input_text):]
            outs.append(output_text)
        temp = {'instruction': instruction, 'input': input, 'responses': outs}
        result.append(temp)

    with open(args.output, 'w') as f:
        json.dump(result, f)      
                  
if __name__ == '__main__':
    main()