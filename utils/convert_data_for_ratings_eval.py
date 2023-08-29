import json
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input", type = str, default = 'sample_generation.json')
parser.add_argument("--output", type = str, default = 'sample_generation_ratings.csv')
args = parser.parse_args()

def main():

    filename = args.input

    with open(filename, 'r') as f:
        data = json.load(f)
        
    instructions = []
    responses    = []
    
    for example in tqdm(data):
        if example['dataset'] != 'selfinstruct':
            instructions.append(example['instruction'])
            responses.append(example['output'])
    df_data = {'instruction': instructions, 'input': len(instructions) * [""], 'response': responses}
    df = pd.DataFrame(df_data)
    df.to_csv(args.output, index = False)

if __name__ == '__main__':
    main()