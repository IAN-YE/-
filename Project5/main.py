import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset, random_split

import torch.optim as optim
import torch.nn as nn
from utils import build_dataset
from model import MultiMidal_concat, MultiMidal_sum, MultiMidal_bilinear
from picture_only import model as pic
from train import train, test
from attention_model import Net
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='model')
    parser.add_argument('--lr', type=float, help='learning_rate')

    args = parser.parse_args()

    picture_model = pic.ResNet(pic.Residual, [1, 1, 1, 1], 3)
    pic_new_model = torch.nn.Sequential(*(list(picture_model.children())[:-1]))

    path = 'text_only/out.txt'
    pretrained_path = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)

    images, input_ids, token_type_ids, attention_mask, labels = build_dataset(path, tokenizer, 64)
    dataset = TensorDataset(images, input_ids, attention_mask, token_type_ids, labels)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    batch_size = 32

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size
    )

    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    if args.model == 'sum':
        model = MultiMidal_sum(pic_new_model, pretrained_path, 768, 3)
    elif args.model == 'concat':
        model = MultiMidal_concat(pic_new_model, pretrained_path, 768, 3)
    elif args.model == 'bilinear':
        model = MultiMidal_bilinear(pic_new_model, pretrained_path, 768, 3)
    else:
        model = Net(pic_new_model, pretrained_path, 3)

    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')

    device = torch.device("cuda:0" if train_on_gpu else "cpu")
    print('using device:', device)

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    train_loss_e, train_acc_e, test_loss_e, test_acc_e = [], [], [], []
    best_acc = 0
    for epoch in range(0, 10):
        print("eopch:{}".format(epoch + 1))
        train_loss, train_acc = train(train_dataloader, model, optimizer, learning_rate=1e-5, device=device)
        test_loss, test_acc = test(validation_dataloader, model, device=device)

        train_loss_e.append(train_loss)
        train_acc_e.append(train_acc)
        test_loss_e.append(test_loss)
        test_acc_e.append(test_acc)

        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), 'best_text.pt')

        print("train_loss:{},train_acc:{}".format(train_loss, train_acc))
        print("test_loss:{},test_acc:{}".format(test_loss, test_acc))


