import process
import MLP
import torch.utils.data as Data
import torchvision.utils
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD,Adam
import seaborn as sns
from sklearn.metrics import accuracy_score

#Checking GPU is used
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")
print('using device:', device)

# Constant to control how frequently we print train loss
print_every = 100

x_train, y_train, x_to_test = process.process()

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=123)

X_train_t = torch.from_numpy(X_train.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_test_t = torch.from_numpy(X_test.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.int64))

X_to_test = torch.from_numpy(x_to_test.astype(np.float32))

# torchsummary.summary(X_train_t, input_size=(3,128,128), device='cuda')

train_data = Data.TensorDataset(X_train_t, y_train_t)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=True,
)

mlpt = MLP.MLP()
paras = list(mlpt.parameters())

optimizer = torch.optim.Adam(mlpt.parameters(), lr=0.005)
loss_func = nn.CrossEntropyLoss()

tb = SummaryWriter()
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)

tb.add_image('texts', grid)
tb.add_graph(mlpt, images)

for epoch in range(10):
    train_loss = 0
    test_accuracy = 0
    for step, (b_x, b_y) in enumerate(train_loader):
        output = mlpt(b_x)
        train_loss = loss_func(output, b_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_output = mlpt(X_train_t)
        _, pre_train = torch.max(train_output, 1)
        output = mlpt(X_test_t)
        _, pre_lab = torch.max(output, 1)
        print(pre_lab)
        train_accuracy = accuracy_score(y_train_t,pre_train)
        test_accuracy = accuracy_score(y_test_t, pre_lab)
        print("epoch:{},train_loss:{},train_accuracy{},test_accuracy:{}".format(epoch, train_loss, train_accuracy, test_accuracy))
        tb.add_scalar('Loss', train_loss, step)
        tb.add_scalar('Accuracy', test_accuracy, step)


y_pre = mlpt(X_to_test)
_, pre_lab = torch.max(y_pre, 1)
pre_lab = pre_lab.tolist()
with open('res.txt', 'w') as f:
    f.write('id, pred\n')
    for data in range(len(pre_lab)):
        f.write('{}, {}\n'.format(data, pre_lab[data]))