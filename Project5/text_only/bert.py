import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModelForMaskedLM

class Model(nn.Module):
    def __init__(self, bert_path, hidden_size, num_classes):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=inputs_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        output = self.fc(output['pooler_output'])
        return output
