import model
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train(train_loader, net, learning_rate, device, dropput=0, optim=None):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    train_loss = 0
    train_accuracy = 0
    train_loss_e = 0

    for step, (b_x, b_y) in enumerate(tqdm(train_loader)):
        b_x, b_y = b_x.to(device), b_y.to(device)

        output = net(b_x)
        train_loss = loss_func(output, b_y)
        # print(loss_func(output, b_y))
        train_loss_e += train_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_output = net(b_x)
        _, pre_train = torch.max(train_output, 1)
        train_accuracy += accuracy_score(b_y.cpu(), pre_train.cpu())

    return train_loss_e / len(train_loader), train_accuracy / len(train_loader)


def test(test_loader, net, device):
    net.eval()
    test_loss = 0
    test_accuracy = 0
    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():
        for step, (t_x, t_y) in enumerate(test_loader):
            t_x, t_y = t_x.to(device), t_y.to(device)

            output = net(t_x)
            test_loss += loss_func(output, t_y)
            _, pre_test = torch.max(output, 1)
            test_accuracy += accuracy_score(t_y.cpu(), pre_test.cpu())

    return test_loss / len(test_loader), test_accuracy / len(test_loader)


def run(train_dataset, test_dataset, learning_rate=1e-3, dropout=0):
    best_acc = 0
    net = model.ResNet(model.Residual, [1, 1, 1, 1], 3)

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')

    device = torch.device("cuda" if train_on_gpu else "cpu")
    print('using device:', device)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True, num_workers=1)

    net.to(device)

    train_loss_e, train_acc_e, test_loss_e, test_acc_e = [], [], [], []
    for epoch in range(10):
        print("eopch:{}".format(epoch + 1))
        train_loss, train_acc = train(train_loader, net, learning_rate=learning_rate, device=device)
        test_loss, test_acc = test(test_loader, net, device=device)

        train_loss_e.append(train_loss.cpu().detach().numpy())
        train_acc_e.append(train_acc)
        test_loss_e.append(test_loss.cpu().detach().numpy())
        test_acc_e.append(test_acc)

        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(net.state_dict(), 'best_vis.pt')

        print("train_loss:{},train_acc:{}".format(train_loss, train_acc))
        print("test_loss:{},test_acc:{}".format(test_loss, test_acc))

    return train_loss_e, train_acc_e, test_loss_e, test_acc_e

