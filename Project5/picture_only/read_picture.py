import cv2
import chardet
import os
import re
import json
from pandas import json_normalize
import numpy as np

def read(path, label):
	texts = []
	labels = []
	map = {"negative":0, "neutral":1, "positive":2}

	width_new, height_new = (224, 224)

	for index, row in label.iterrows():
		file = str(row['id']) + '.jpg'
		img = cv2.imread(path + file)
		labels.append(map[row["tag"]])

		width, height, channel = img.shape

		if width < width_new or height < height_new:
			dim_diff = np.abs(height - width)

			pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
			pad = (0, 0, pad1, pad2) if width <= height else (pad1, pad2, 0, 0)
			top, bottom, left, right = pad

			pad_value = 0
			img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, pad_value)
			img_new = cv2.resize(img_pad, (224, 224), interpolation=cv2.INTER_AREA)

		else:
			img_new = img[width // 2 - width_new // 2 : width // 2 + width_new // 2,
					  height // 2 - height_new // 2 : height // 2 + height_new // 2, :]

		if img_new.shape[0] != 224 or img_new.shape[1] != 224:
			print("wrong picture:{}, shape{}, new shape{}".format(row['id'], img.shape, img_new.shape))

		texts.append(img_new)

	np.save(file="picture.npy", arr=np.array(texts))
	np.save(file="label.npy", arr=np.array(labels))

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

	# label = read_label()
	# read(path, label)

	data = np.load(file='picture.npy')
	labels = np.load(file="label.npy")

	print(labels)
