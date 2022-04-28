import pandas as pd
import numpy as np

def mask(name,phone,identity):
    name_mask, phone_mask, identity_mask = None, None, None
    if len(name) == 2:
        name_mask = name[0] +"*"
    elif len(name) == 3:
        name_mask = name[:1] + "**"
    else:
        name_mask = name[:2] + "**"

    phone_mask = phone[:3] + "****" + phone[7:]
    identity_mask = identity[:6] + "********" + identity[14:]

    return [name_mask,phone_mask,identity_mask]

data = pd.read_csv(r'Exp 6_problem2.csv')
res = []

for index,row in data.iterrows():
    tmp = mask(str(row['姓名']),str(row['手机号']),str(row['身份证号']))
    res.append(tmp)

print(pd.DataFrame(res))