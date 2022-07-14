import torch
import torch.nn as nn
import numpy as np
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer
import logging

logger = logging.getLogger(__name__)

class Codet5(nn.Module):
    def __init__(self, path):
        super(Codet5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(path)

    def forward(self, input_ids=None, attention_mask=None, labels=None, decoder_attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             labels=labels, decoder_attention_mask=decoder_attention_mask)
        return outputs
