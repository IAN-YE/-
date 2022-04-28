import LeNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

train_dataset = datasets.MNIST(root = 'data/', train = True,transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = 'data/', train = False,transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
images, labels = next(iter(train_loader))

if __name__ == '__main__':
    lenet = LeNet.Model()
    optimizer = torch.optim.Adam(lenet.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(10):
        train_loss_e = 0
        train_accuracy = 0
        test_accuracy = 0
        #lr = 0.95**epoch*lr

        for step, (b_x, b_y) in enumerate(train_loader):
            output = lenet(b_x)
            train_loss = loss_func(output, b_y)
            train_loss_e += train_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_output = lenet(b_x)
            _, pre_train = torch.max(train_output, 1)
            output = lenet(X_test_t)
            _, pre_lab = torch.max(output, 1)
            #print(pre_lab)
            train_accuracy += accuracy_score(y_train_t, pre_train)
            test_accuracy += accuracy_score(y_test_t, pre_lab)

        print("{},{},{},{}".format(epoch, train_loss_e/len(train_loader),
               train_accuracy/len(train_loader),test_accuracy/len(train_loader)))
        # tb.add_scalar('Loss', (, epoch)
        # tb.add_scalar('Accuracy', (, epoch)