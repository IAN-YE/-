import torch
import json
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def build_dataset(path, tokenizer, max_seq_length):
    input_ids, token_type_ids, attention_mask, labels = [], [], [], []
    mapping = {'negative': 0, 'positive': 1, 'neutral': 2}
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        data = json.loads(line.rstrip('\r\n'))
        label = data["tag"]
        text = data["text"]
        token = tokenizer(text, max_length=max_seq_length, truncation=True,
                          padding='max_length', add_special_tokens=True)
        input_ids.append(token.input_ids)
        token_type_ids.append(token.token_type_ids)
        attention_mask.append(token.attention_mask)
        labels.append(int(mapping[label]))

    input_ids = torch.tensor(input_ids).long()
    token_type_ids = torch.tensor(token_type_ids).long()
    attention_mask = torch.tensor(attention_mask).long()
    labels = torch.Tensor(labels).long()

    return input_ids, token_type_ids, attention_mask, labels

def train(train_loader, model, optimizer, device, learning_rate=1e-5, dropput=0, optim=None):
    model.train()
    total_train_loss = 0
    train_accuracy = 0
    criterion = nn.CrossEntropyLoss()

    for step, batch in enumerate(tqdm(train_loader)):
        images = batch[0].float().to(device)
        input_ids = batch[1].to(device)
        input_mask = batch[2].to(device)
        input_token_type_ids = batch[3].to(device)
        labels = batch[4].to(device)

        model.zero_grad()

        output = model(input_ids, input_mask, input_token_type_ids, images)

        loss = criterion(output, labels)
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _, pre_lab = torch.max(output, 1)
        # print(pre_lab)
        train_accuracy += accuracy_score(labels.cpu().numpy(), pre_lab.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("{}".format(train_accuracy / len(train_loader)))

    return total_train_loss / len(train_loader), train_accuracy / len(train_loader)

def test(validation_dataloader, model, device):
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    criterion = nn.CrossEntropyLoss()

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        images = batch[0].float().to(device)
        input_ids = batch[1].to(device)
        input_mask = batch[2].to(device)
        input_token_type_ids = batch[3].to(device)
        labels = batch[4].to(device)

        with torch.no_grad():
            output = model(input_ids, input_mask, input_token_type_ids, images)

        loss = criterion(output, labels)

        total_eval_loss += loss.item()

        _, pre_out = torch.max(output, 1)

        total_eval_accuracy += accuracy_score(pre_out.cpu().numpy(), labels.cpu().numpy())

    return total_eval_loss / len(validation_dataloader), total_eval_accuracy / len(validation_dataloader)


