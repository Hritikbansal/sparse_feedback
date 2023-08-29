import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--input", type = str, default = 'sample_generation.json')
parser.add_argument("--output", type = str, default = 'rankings_sample_generation.csv')

args = parser.parse_args()


def main():

    with open(args.input, 'r') as f:
        data = json.load(f)

    print(len(data))
    instructions = []
    inputs = []
    responses1 = []
    responses2 = []

    for ex in data:
        responses = ex['responses']
        for i in range(len(responses) - 1):
            for j in range(i + 1, len(responses)):                
                instructions.append(ex['instruction'])
                inputs.append(ex['input'])
                responses1.append(responses[i])
                responses2.append(responses[j])

    data = {'instruction': instructions, 'input': inputs, 'response1': responses1, 'response2': responses2}
    df = pd.DataFrame(data)
    print(len(df))
    df.to_csv(args.output, index = False)

if __name__ == "__main__":
    main()