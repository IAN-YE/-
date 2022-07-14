import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel



class Net(nn.Module):
    def __init__(self, pic_model, bert_path, out_class):
        super(Net, self).__init__()
        self.pic_model = pic_model
        self.text_model = BertModel.from_pretrained(bert_path)

        question_features = 768
        vision_features = 768
        glimpses = 2

        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=768*2,
            glimpses=glimpses,
            )

        self.classifier = Classifier(
            in_features=(glimpses * vision_features, question_features),
            mid_features=1024,
            out_features=out_class,
            drop=0.5, )

        self.extra = nn.Linear(28*28, 768)

    def forward(self, inputs_ids, attention_mask, token_type_ids, v):
        q = self.text_model(inputs_ids, attention_mask, token_type_ids)['pooler_output'] # [batch, 768]
        v = self.pic_model(v)
        v = v.view(v.shape[0],v.shape[1],28*28)
        v = self.extra(v)

        a = self.attention(v, q)  # [batch, 36, num_glimpse]
        v = apply_attention(v.transpose(1, 2), a)  # [batch, 2048 * num_glimpse]
        answer = self.classifier(v, q)

        return answer


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses):
        super(Attention, self).__init__()
        self.lin_v = FCNet(v_features, mid_features, activate='relu')  # let self.lin take care of bias
        self.lin_q = FCNet(q_features, mid_features, activate='relu')
        self.lin = FCNet(mid_features, glimpses)

    def forward(self, v, q):
        """
        v = batch, num_obj, dim
        q = batch, dim
        """
        v = self.lin_v(v)
        q = self.lin_q(q)
        batch, num_obj, _ = v.shape
        _, q_dim = q.shape
        q = q.unsqueeze(1).expand(batch, num_obj, q_dim)

        x = v * q
        x = self.lin(x)  # batch, num_obj, glimps
        x = F.softmax(x, dim=1)
        return x


def apply_attention(input, attention):
    """
    input = batch, dim, num_obj
    attention = batch, num_obj, glimps
    """
    batch, dim, _ = input.shape
    _, _, glimps = attention.shape
    x = input @ attention  # batch, dim, glimps
    assert (x.shape[1] == dim)
    assert (x.shape[2] == glimps)
    return x.view(batch, -1)

class Classifier(nn.Module):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.lin11 = FCNet(in_features[0], mid_features, activate='relu')
        self.lin12 = FCNet(in_features[1], mid_features, activate='relu')
        self.lin2 = FCNet(mid_features, mid_features, activate='relu')
        self.lin3 = FCNet(mid_features, out_features, drop=drop)

    def forward(self, v, q):
        #x = self.fusion(self.lin11(v), self.lin12(q))
        x = self.lin11(v) * self.lin12(q)
        x = self.lin2(x)
        x = self.lin3(x)
        return x


class FCNet(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)

        self.drop_value = drop
        self.drop = nn.Dropout(drop)

        # in case of using upper character by mistake
        self.activate = activate.lower() if (activate is not None) else None
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)

        x = self.lin(x)

        if self.activate is not None:
            x = self.ac_fn(x)
        return x

