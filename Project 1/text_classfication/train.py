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

x_train, y_train, x_to_test = process.process()

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=123)

X_train_t = torch.from_numpy(X_train.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_test_t = torch.from_numpy(X_test.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.int64))

train_data = Data.TensorDataset(X_train_t,y_train_t)

train_loader = Data.DataLoader(
    dataset = train_data ,
    batch_size = 64 ,
    shuffle = True,
)

mlpt = MLP.MLP()
optimizer = torch.optim.Adam(mlpt.parameters(),lr=0.05)
loss_func = nn.CrossEntropyLoss()    ###二分类损失函数

####记录训练过程的指标
history1 = hl.History()

####使用Canvas进行可视化
canvas1 = hl.Canvas()
print_step = 25

###对模型进行迭代训练，对所有数据训练epoch轮
for epoch in range(15):
    ##对训练数据的加载器进行迭代计算
    for step, (b_x , b_y) in enumerate(train_loader):
        ##计算每个batch的损失
        _, _, output = mlpt(b_x)   ###MLP在训练batch上的输出
        train_loss = loss_func(output , b_y)   ####二分类交叉熵损失函数
        optimizer.zero_grad()     #####每个迭代的梯度初始化为0
        train_loss.backward()     #####损失的后向传播，计算梯度
        optimizer.step()         #####使用梯度进行优化
        niter = epoch*len(train_loader) + step +1
        if niter % print_step == 0:
            _, _, output = mlpt(X_test_t)
            _, pre_lab = torch.max(output, 1)
            test_accuracy = accuracy_score(y_test_t, pre_lab)
            ## 为history添加epoch,损失和精度
            history1.log(niter, train_loss=train_loss, test_accuracy=test_accuracy)
            print("train_loss{},test_accuracy{}".format(train_loss,test_accuracy))
            print(torch.cuda.is_available())
            ###使用两个图像可视化损失函数和精度
            # with canvas1:
            #     canvas1.draw_plot(history1['train_loss'])
            #     canvas1.draw_plot(history1['test_accuracy'])

