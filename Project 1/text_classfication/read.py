import json

Data = []
f = open('C:/Users/86189/Desktop/当代人工智能/exp1data/train_data.txt', 'r', encoding='utf-8')
for line in f.readlines():
    try:
        Data.append(json.loads(line.rstrip(';\n')))
    except ValueError:
        print("Skipping invalid line {0}".format(repr(line)))

f.close()

Test = []
file = open('C:/Users/86189/Desktop/当代人工智能/exp1data/test.txt', 'r', encoding='utf-8')
for line in file.readlines():
    tmp = {}
    split = line.split(',', 1)
    tmp["id"] = split[0]
    tmp["text"] = split[1]
    Test.append(tmp)

file.close()