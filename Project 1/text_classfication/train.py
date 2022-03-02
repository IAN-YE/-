import process
import MLP
import torch.utils.data as Data
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import SGD,Adam
import seaborn as sns
import hiddenlayer as hl
from sklearn.metrics import accuracy_score

USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)

x_train, y_train, x_to_test = process.process()

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=123)

X_train_t = torch.from_numpy(X_train.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_test_t = torch.from_numpy(X_test.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.int64))

train_data = Data.TensorDataset(X_train_t,y_train_t)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
)

mlpt = MLP.MLP()
paras = list(mlpt.parameters())

optimizer = torch.optim.Adam(mlpt.parameters(), lr=0.005)
loss_func = nn.CrossEntropyLoss()

history1 = hl.History()

canvas1 = hl.Canvas()
print_step = 25

for epoch in range(15):
    for step, (b_x, b_y) in enumerate(train_loader):
        _, _, output = mlpt(b_x)
        train_loss = loss_func(output , b_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        niter = epoch * len(train_loader) + step + 1
        _, _, output = mlpt(X_test_t)
        _, pre_lab = torch.max(output, 1)
        print(pre_lab)
        test_accuracy = accuracy_score(y_test_t, pre_lab)
        ## 为history添加epoch,损失和精度
        history1.log(niter, train_loss=train_loss, test_accuracy=test_accuracy)
        print("epoch:{},train_loss:{},test_accuracy:{}".format(epoch, train_loss, test_accuracy))

        # with canvas1:
        #     canvas1.draw_plot(history1['train_loss'])
        #     canvas1.draw_plot(history1['test_accuracy'])

