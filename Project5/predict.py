import json
import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset, random_split

from utils import read_img
from picture_only import model as pic
from model import MultiMidal_bilinear, MultiMidal_sum
import numpy as np

if __name__ == '__main__':
    path = 'test.txt'
    pretrained_path = 'bert-base-multilingual-cased'

    picture_model = pic.ResNet(pic.Residual, [1, 1, 1, 1], 3)
    pic_new_model = torch.nn.Sequential(*(list(picture_model.children())[:-1]))

    model = MultiMidal_sum(pic_new_model, pretrained_path, 768, 3)
    model.load_state_dict(torch.load('best.pt'))

    tokenizer = BertTokenizer.from_pretrained(pretrained_path)

    doc = open('result.txt', 'w')
    mapping = {0: 'negative', 1: 'positive', 2: 'neutral'}
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        data = json.loads(line.rstrip('\r\n'))
        label = data["tag"]
        text = data["text"]
        id = data["id"]

        image = read_img('data/data/data/data/' + str(id) + '.jpg')

        token = tokenizer(text, max_length=64, truncation=True,
                          padding='max_length', add_special_tokens=True)

        input_ids = torch.tensor([token.input_ids]).long()
        token_type_ids = torch.tensor([token.token_type_ids]).long()
        attention_mask = torch.tensor([token.attention_mask]).long()
        image = torch.from_numpy(np.array(image)).unsqueeze(0)
        image = image.permute(0, 3, 1, 2).float()

        output = model(input_ids, attention_mask, token_type_ids, image)

        _, pre_out = torch.max(output, 1)

        print("{},{}".format(id, mapping[int(pre_out)]), file=doc)






