import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=strides),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        if strides != 1 or in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None


    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2(output)

        if self.conv3:
            output = output + self.conv3(x)

        output = F.relu(output)

        return output

class ResNet(nn.Module):
    def __init__(self, block, num_block, out_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.block1 = self.Resnet_Block(16, num_block[0], stride=1)
        self.block2 = self.Resnet_Block(32, num_block[1], stride=2)
        self.block3 = self.Resnet_Block(64, num_block[2], stride=2)

        self.outlayer = nn.Linear(64, out_classes)

    def Resnet_Block(self, out_channels, num_res, stride):
        layers = []
        for i in range(num_res):
            if i == 0:
                layers.append(Residual(self.in_channels, out_channels, stride))
            else:
                layers.append(Residual(self.in_channels, out_channels, 1))

        self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)            #[1,1,28,28]
        x = self.block1(x)           #[1,1,28,28]
        x = self.block2(x)           #[1,1,14,14]
        x = self.block3(x)           #[1,1,7,7]
        x = F.avg_pool2d(x, 7)
        output = self.outlayer(x.view(x.size(0), -1))

        return output




