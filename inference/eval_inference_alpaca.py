import csv
import json
import torch
import argparse
import transformers
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--device_id", type = int, default = 3)
parser.add_argument("--temp", type = float, default = 1.0)
parser.add_argument("--input_data", type = str, default = 'davinci003.json')
parser.add_argument("--save_generations", type = str, default = 'alpaca7b.json')
parser.add_argument("--model_path", type = str, default = "alpaca7b")

args = parser.parse_args()

model_path = args.model_path

model = transformers.AutoModelForCausalLM.from_pretrained(model_path, 
                                                            device_map = {"": torch.device(f"cuda:{args.device_id}")},
                                                            torch_dtype = torch.float16,
                                                            low_cpu_mem_usage=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def main():

    datasets = {'helpful_base', 'vicuna', 'koala', 'oasst'}
    
    with open(args.input_data, 'r') as f:
        data = json.load(f)

    result = []
    for example in tqdm(data):
        if example['dataset'] in datasets:
            response = {}
            response['dataset'] = example['dataset']
            response['generator'] = 'alpaca_7b'
            instruction = example['instruction']
            response['instruction'] = instruction
            outs = []
            input_text = PROMPT_DICT['prompt_no_input'].format(instruction = instruction)
            inputs = tokenizer(input_text, return_tensors="pt")
            for _ in range(8):
                output_texts = model.generate(inputs=inputs.input_ids.to(f"cuda:{args.device_id}"), max_new_tokens = 200, num_return_sequences=8, temperature = args.temp, do_sample=True)
                output_texts = tokenizer.batch_decode(output_texts, skip_special_tokens=True)
                for output_text in output_texts:
                    output_text = output_text[len(input_text):]
                    outs.append(output_text)
            response['outputs'] = outs
            result.append(response)
            with open(args.save_generations, 'w') as f:
                json.dump(result, f)

if __name__ == '__main__':
    main()

'''
alpaca_eval --model_outputs model_outputs/absolute_model_2.json --annotators_config chatgpt_fn --reference_outputs model_outputs/alpaca_eval_random_alpaca_7b.json 
'''