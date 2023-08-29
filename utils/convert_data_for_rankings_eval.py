import json
import random
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--davinci_file", type = str, default = 'davinci.json')
parser.add_argument("--alpaca_file", type = str, default = 'alpaca.json')
parser.add_argument("--output", type = str, default = 'output.csv')
args = parser.parse_args()



def main():

    with open(args.davinci_file, 'r') as f:
        davinci_data = json.load(f)

    with open(args.alpaca_file, 'r') as f:
        alpaca_data = json.load(f)

    instructions = []
    davinci_outputs = []
    alpaca_outputs = []

    total = 0
    for example in tqdm(alpaca_data):
        instruction = example['instruction']
        alpaca_output = example['output']

        for davinci_example in davinci_data:
            if davinci_example['instruction'] == instruction:
                davinci_output = davinci_example['output']
                total += 1

        instructions.append(instruction)
        davinci_outputs.append(davinci_output)
        alpaca_outputs.append(alpaca_output)

    print(total)
    all_data = {'instruction': instructions, 'input': len(instructions) * [""], 'response1': davinci_outputs, 'response2': alpaca_outputs}
    df = pd.DataFrame(all_data)
    df.to_csv(args.output, index = False)
    print(len(df))


if __name__ == "__main__":
    main()