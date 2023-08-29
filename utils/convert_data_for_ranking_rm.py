import json
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument("--input", type = str, default = 'sample_generation.json')
parser.add_argument("--output", type = str, default = 'rankings_sample_generation.csv')

args = parser.parse_args()

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


def create_instruction(instruction, input):
    if str(input) == 'nan' or input == "":
        instruction = PROMPT_DICT['prompt_no_input'].format(instruction = instruction)
    else:
        instruction = PROMPT_DICT['prompt_input'].format(instruction = instruction, input = input)
    return instruction

df = pd.read_csv(args.input, names = ['instruction', 'input', 'output1', 'output2', 'ranking'])
df = df.drop_duplicates()

result = defaultdict(lambda: defaultdict(list))

for i in tqdm(range(len(df))):
    instruction, input = df.iloc[i]['instruction'], df.iloc[i]['input']
    instruction = create_instruction(instruction, input)
    output_1, output_2 = df.iloc[i]['output1'], df.iloc[i]['output2']
    feedback = df.iloc[i]['ranking']
    if '(a)' == feedback:
        result[instruction]['sentences'].append([output_1, output_2])
    elif '(b)' == feedback:
        result[instruction]['sentences'].append([output_2, output_1])

print(len(result))

with open(args.output, 'w') as f:
    json.dump(result, f)
