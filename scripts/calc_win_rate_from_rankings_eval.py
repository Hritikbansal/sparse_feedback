import argparse
import pandas as pd
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser()

parser.add_argument('--input', type = str, default = 'feedback_ratings_davinci.json')

args = parser.parse_args()

df = pd.read_csv(args.input, names = ['ins', 'input', 'response1', 'response2', 'feedback'])
print(len(df))

feedback = df['feedback'].tolist()
## (b) is usually the model being aligned (a) corresponds to the reference model
win = [1 if x == '(b)' else 0.5 if x == 'equal' else 0 for x in feedback]
win = np.array(win)

print(f"Mean: {100 * win.mean()}")
print(f"95% CI: {196 * stats.sem(win)}")