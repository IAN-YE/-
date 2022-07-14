import chardet
import os
import re
import json
from pandas import json_normalize

def read(path):
	text = None

	try:
		f = open(path, 'rb')  # 打开txt文档
		r = f.read()  # 读取
		f_charInfo = chardet.detect(r)  # 获取文本编码信息
		# print(f_charInfo)
		# print(f_charInfo['encoding'])  # 取得文本格式
		# print(path)
		if f_charInfo['encoding'] == 'GB2312' or f_charInfo['encoding'] == 'Big5':
			encoding = 'GBK'
		else:
			encoding = f_charInfo['encoding']
		f.close()
	except:
		if f:
			f.close()
			print('err')

	f = open(path, encoding=encoding)
	text = f.read().rstrip("\n")

	return text

def read_label():
    Test = []
    f = open('../data/实验五数据/实验五数据/train.txt', 'r',encoding='utf-8')
    for line in f.readlines():
        tmp = {}
        split = line.split(',', 1)
        tmp["id"] = split[0]
        tmp["tag"] = split[1].rstrip()
        Test.append(tmp)

    f.close()

    df = json_normalize(Test)
    df.drop(axis=0, index=0, inplace=True)
    return df

if __name__=='__main__':
	path = '../data/实验五数据/实验五数据/data/'
	pattern = re.compile(r'\d+')
	files = os.listdir(path)

	label = read_label()

	doc = open('out.txt', 'w')
	texts = []

	for index, row in label.iterrows():
		file = str(row['id']) + '.txt'
		out = dict()
		#print(file)
		text = read(path + file)
		if text == None:
			print(file)
		out["id"] = row["id"]
		out["tag"] = row["tag"]
		out["text"] = text
		print(json.dumps(out), file=doc)

	doc.close()

	# file = str(70) + '.txt'
	# text = read(path + file)
	# print(text)




