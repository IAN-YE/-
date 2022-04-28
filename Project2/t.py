import numpy as np
import torch
from torch import nn


def corr2d(X, K):
    '''
    X --> (B, I, H, W) where B = batch size, I = in_channel, H = height of feature map, W = width of feature map
    K --> (O, I, h, w) where O = out_channel, I = in_channel, h = height of kernel, w = width of kernel
    你需要实现一个Stride为1，Padding为0的窄卷积操作
    Y的大小应为(B, O, H-h+1, W-w+1)
    '''
    # =============
    # todo: 请根据以上提示补全代码
    B, I, H, W = X.shape
    O, I, h, w = K.shape

    outH = H - h + 1
    outW = W - w + 1
    Y = torch.zeros((B, O, outH, outW))
    k_row = K.reshape((O, I * h * w))
    x_col = torch.zeros((I * h * w, outH * outW))

    for index in range(B):
        neuron = 0
        for i in range(0, H - h + 1):
            for j in range(0, W - w + 1):
                x_col[:, neuron] = X[index, :, i:i + h, j:j + w].reshape(I * h * w)
                neuron += 1
            Y[index] = (k_row @ x_col).reshape(O, outH, outW)

    # =============
    return Y


class Conv2D(nn.Module):
    def __init__(self, out_channels, in_channels, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size[0], kernel_size[1])))
        self.bias = nn.Parameter(torch.randn((out_channels)))

    def forward(self, X):
        Y = corr2d(X, self.weight) + self.bias.view(1, -1, 1, 1)
        return Y


class MaxPool2D(nn.Module):
    def __init__(self, pool_size):
        super(MaxPool2D, self).__init__()
        self.pool_size = pool_size

    def forward(self, X):
        '''
        X --> (B, I, H, W) where B = batch size, I = in_channel, H = height of feature map, W = width of feature map
        K --> (h, w) where h = height of kernel, w = width of kernel
        你需要利用以上pool_size实现一个最大汇聚层的前向传播，汇聚层的子区域间无覆盖
        Y的大小应为(B, I, H/h, W/w)
        '''

        # =============
        # todo: 请根据以上提示补全代码
        B, I, H, W = X.shape
        h = self.pool_size[0]
        w = self.pool_size[1]
        outH = int(H / h)
        outW = int(W / h)
        Y = torch.zeros((B,I,outH,outW))

        for index in range(B):
            out_col = torch.zeros((I, outH * outW))
            neuron = 0
            for i in range(0, H - h + 1, h):
                for j in range(0, W - w + 1, w):
                    pool_region = X[index, :, i:i + h, j:j + w].reshape(I, h * w)
                    out_col[:, neuron] = pool_region.detach().max(axis=1)
                    neuron += 1
            Y[index] = out_col.reshape(I, outH, outW)

        # =============
        return Y

class ImageCNN(nn.Module):
    def __init__(self, num_outputs, in_channels, out_channels, conv_kernel, pool_kernel):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2D(out_channels, in_channels, conv_kernel),
            nn.ReLU()
        )
        self.pool1 = MaxPool2D(pool_kernel)
        self.linear = nn.Linear(16 * 5 * 5, num_outputs)

    def forward(self, feature_map):
        b = feature_map.size()[0]
        feature_map = self.conv1(feature_map)
        feature_map = self.pool1(feature_map)
        outputs = self.linear(feature_map.reshape(b, -1))

        return outputs

