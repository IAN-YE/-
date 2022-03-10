import json
from pandas import json_normalize


def read_train_data():
    Data = []
    f = open('../exp1data/train_data.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        try:
            Data.append(json.loads(line.rstrip('\r\n')))
        except ValueError:
            print("Skipping invalid line {0}".format(repr(line)))

    f.close()

    df = json_normalize(Data)

    return df

def read_test_data():
    Test = []
    f = open('../exp1data/test.txt', 'r',
                encoding='utf-8')
    for line in f.readlines():
        tmp = {}
        split = line.split(',', 1)
        tmp["id"] = split[0]
        tmp["text"] = split[1]
        Test.append(tmp)

    f.close()

    df = json_normalize(Test)
    df.drop(axis=0, index=0, inplace=True)
    return df
