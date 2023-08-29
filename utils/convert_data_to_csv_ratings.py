import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--input", type = str, default = 'sample_generation.json')
parser.add_argument("--output", type = str, default = 'sample_generation_ratings.csv')

args = parser.parse_args()


def main():

    with open(args.input, 'r') as f:
        data = json.load(f)

    print(len(data))
    instructions = []
    inputs = []
    responses = []

    for ex in data:
        for response in ex['responses']:
            instructions.append(ex['instruction'])
            inputs.append(ex['input'])
            responses.append(response)

    data = {'instruction': instructions, 'input': inputs, 'response': responses}
    df = pd.DataFrame(data)
    print(len(df))
    df.to_csv(args.output, index = False)

if __name__ == "__main__":
    main()