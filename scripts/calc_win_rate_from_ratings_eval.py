import argparse
import pandas as pd
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser()

parser.add_argument('--input1', type = str, default = 'feedback_ratings_davinci.json')
parser.add_argument('--input2', type = str, default = 'feedback_ratings_model.json')

args = parser.parse_args()

df1 = pd.read_csv(args.input1, names = ['ins', 'inp', 'out', 'score'])
df1 = df1.drop_duplicates(subset=['ins'])
print(len(df1))

df2 = pd.read_csv(args.input2, names = ['ins', 'inp', 'out', 'score'])
df2 = df2.drop_duplicates(subset=['ins'])
print(len(df2))

df_merge = pd.merge(df1, df2, on = ['ins'], how = 'inner')
print(len(df_merge))

win = []
total = 0

for j in range(len(df_merge)):
    if df_merge.iloc[j]['score_x'] == df_merge.iloc[j]['score_y']:
        win.append(0.5)
    elif  df_merge.iloc[j]['score_x'] < df_merge.iloc[j]['score_y']:    
        win.append(1)
    else:
        win.append(0)
    total += 1

win = np.array(win)
print(f"Mean: {100 * win.mean()}")
print(f"95% CI: {196 * stats.sem(win)}")