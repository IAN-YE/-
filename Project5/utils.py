import torch
import json
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import cv2
import numpy as np

def read_img(path):
    width_new, height_new = (224, 224)
    img = cv2.imread(path)

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
        img_new = img[width // 2 - width_new // 2: width // 2 + width_new // 2,
                      height // 2 - height_new // 2: height // 2 + height_new // 2, :]

    if img_new.shape[0] != 224 or img_new.shape[1] != 224:
        print("wrong picture:{}, shape{}, new shape{}".format(path, img.shape, img_new.shape))

    return img_new

def build_dataset(path, tokenizer, max_seq_length):
    images, input_ids, token_type_ids, attention_mask, labels = [], [], [], [], []
    mapping = {'negative': 0, 'positive': 1, 'neutral': 2, 'null': 3}
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        data = json.loads(line.rstrip('\r\n'))
        label = data["tag"]
        text = data["text"]
        id = data["id"]

        image = read_img('data/data/data/data/' + str(id) + '.jpg')

        token = tokenizer(text, max_length=max_seq_length, truncation=True,
                          padding='max_length', add_special_tokens=True)
        input_ids.append(token.input_ids)
        token_type_ids.append(token.token_type_ids)
        attention_mask.append(token.attention_mask)
        labels.append(int(mapping[label]))
        images.append(image)

    input_ids = torch.tensor(input_ids).long()
    token_type_ids = torch.tensor(token_type_ids).long()
    attention_mask = torch.tensor(attention_mask).long()
    labels = torch.Tensor(labels).long()
    images = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2)

    return images, input_ids, token_type_ids, attention_mask, labels
