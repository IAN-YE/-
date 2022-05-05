import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2,stride=2)
        )

        self.outlayer = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self,x):
        x  = self.net(x)
        output = self.outlayer(x.view(x.size(0), -1))
        return output