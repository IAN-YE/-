import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Sigmoid(),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(400,10)
        )

    def forward(self,x):
        output = self.net(x)
        return output