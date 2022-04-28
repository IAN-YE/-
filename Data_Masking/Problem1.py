import pandas as pd
import numpy as np

def process(input):
    res = dict()
    count = 0
    for x in input:
        res[x]  = count
        count += 1
    return res

data = pd.read_csv(r'Exp 6_problem1.csv')
data.insert(loc=0,column='new_ID',value=-1)

map = process(data['ID'].unique())

for index,row in data.iterrows():
    data.at[index, 'new_ID'] = map[row['ID']]

print(data[['new_ID','事件']])