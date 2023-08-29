import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--ratings_csv', type = str, default = 'data/alpaca_generation/feedback_ratings_sample_generation.csv')
parser.add_argument('--rankings_csv', type = str, default = 'data/alpaca_generation/feedback_rankings_sample_generation.csv')

args = parser.parse_args()


def get_absolute_ranking(instruction, input, output_1, output_2, output_dict):
    key1 = f"{instruction}-{input}-{output_1}"
    if key1 in output_dict:
        score1 = output_dict[key1]
    else:
        return None, None, "none"
    key2 = f"{instruction}-{input}-{output_2}"
    if key2 in output_dict:
        score2 = output_dict[key2]
    else:
        return None, None, "none"

    if score1 == score2:
        return score1, score2, "equal"
    elif score1 > score2:
        return score1, score2, "respA"
    else:
        return score1, score2, "respB"

def main():

    df1 = pd.read_csv(args.ratings_csv, names = ['instruction', 'input', 'output', 'rating'])
    df1 = df1.drop_duplicates(subset = ['instruction', 'input', 'output'])
    
    df2 = pd.read_csv(args.rankings_csv, names = ['instruction', 'input', 'output1', 'output2', 'ranking'])
    df2 = df2.drop_duplicates(subset = ['instruction', 'input', 'output1', 'output2'])

    print(len(df1), len(df2))
    output_dict = {}

    for j in range(len(df1)):
        key = f"{df1.iloc[j]['instruction']}-{df1.iloc[j]['input']}-{df1.iloc[j]['output']}"
        output_dict[key] = df1.iloc[j]['rating']

    print(len(output_dict))

    comparison_dict = {"equal":[0, 0, 0, 0], "respA": [0, 0, 0, 0], "respB": [0, 0, 0, 0]}
    total = 0

    for i in tqdm(range(len(df2))):
        output_1, output_2 = df2.iloc[i]['output1'], df2.iloc[i]['output2']
        ranking = df2.iloc[i]['ranking']
        score1, score2, absolute_ranking = get_absolute_ranking(df2.iloc[i]['instruction'], df2.iloc[i]['input'], output_1, output_2, output_dict)
        if absolute_ranking == "none":
            continue
        if ranking == "equal":
            comparison_dict[absolute_ranking][0] += 1
            comparison_dict[absolute_ranking][3] += 1
            total += 1
        elif '(a)' == ranking:
            comparison_dict[absolute_ranking][1] += 1
            comparison_dict[absolute_ranking][3] += 1
            total += 1
        elif '(b)' == ranking:
            comparison_dict[absolute_ranking][2] += 1
            comparison_dict[absolute_ranking][3] += 1
            total += 1
    consistency = (comparison_dict['equal'][0] + comparison_dict['respA'][1] + comparison_dict['respB'][2])/total
    print(f"Consistency: {100 * consistency}")
    
if __name__ == "__main__":
    main()