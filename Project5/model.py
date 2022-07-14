import torch
import torch.nn as nn
from picture_only import model as pic
from text_only import bert as text
from transformers import BertModel
import torch.nn.functional as F

class MultiMidal_concat(nn.Module):
    def __init__(self, pic_model, bert_path, hidden, out_class):
        super(MultiMidal_concat, self).__init__()
        self.pic_model = pic_model
        self.text_model = BertModel.from_pretrained(bert_path)

        self.fc1 = nn.Linear(512 * 7 * 7, hidden)

        self.fc2 = nn.Linear(hidden * 2, out_class)

    def forward(self,inputs_ids, attention_mask, token_type_ids, pictures):
        pic = self.pic_model(pictures)
        pic = F.avg_pool2d(pic, 4)
        pic_emb = self.fc1(pic.view(pic.size(0), -1))

        text_emb = self.text_model(inputs_ids,attention_mask,token_type_ids)['pooler_output']
        x = torch.cat([pic_emb, text_emb], dim=1)
        out = self.fc2(x)

        return out

class MultiMidal_sum(nn.Module):
    def __init__(self, pic_model, bert_path, hidden, out_class):
        super(MultiMidal_sum, self).__init__()
        self.pic_model = pic_model
        self.text_model = BertModel.from_pretrained(bert_path)

        self.fc1 = nn.Linear(512 * 7 * 7, hidden)

        self.fc2 = nn.Linear(hidden, 250)
        self.out = nn.Linear(250, out_class)

    def forward(self, inputs_ids, attention_mask, token_type_ids, pictures):
        pic = self.pic_model(pictures)
        pic = F.avg_pool2d(pic, 4)
        pic_emb = self.fc1(pic.view(pic.size(0), -1))

        text_emb = self.text_model(inputs_ids, attention_mask, token_type_ids)['pooler_output']
        x = pic_emb + text_emb
        x = self.fc2(x)
        output = self.out(x)

        return output

class MultiMidal_bilinear(nn.Module):
    def __init__(self, pic_model, bert_path, hidden, out_class):
        super(MultiMidal_bilinear, self).__init__()
        self.pic_model = pic_model
        self.text_model = BertModel.from_pretrained(bert_path)

        self.fc1 = nn.Linear(512 * 7 * 7, hidden)

        self.fc2 = nn.Linear(hidden * hidden, 250)
        self.out = nn.Linear(250, out_class)

    def forward(self, inputs_ids, attention_mask, token_type_ids, pictures):
        pic = self.pic_model(pictures)
        pic = F.avg_pool2d(pic, 4)
        pic_emb = self.fc1(pic.view(pic.size(0), -1))

        text_emb = self.text_model(inputs_ids, attention_mask, token_type_ids)['pooler_output']

        pic_size = pic_emb.size()
        text_size = text_emb.size()

        out_size = list(pic_size)
        out_size[-1] = pic_size[-1] * text_size[-1]  # 特征x和特征y维数之积

        x = pic_emb.view([-1, int(pic_size[-1])])
        y = text_emb.view([-1, int(text_size[-1])])

        out_stack = []

        for i in range(x.size()[0]):
            out_stack.append(torch.ger(x[i], y[i]))  # torch.ger()向量的外积操作
        out = torch.stack(out_stack)  # 将list堆叠成tensor

        out = self.fc2(out.view(out.size(0), -1))

        output = self.out(out)

        return output

