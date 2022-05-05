import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, dropout = 0.5):
        super(AlexNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256*3*3, 1024), nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 10)
        )

    def forward(self,x):
        output = self.net(x)
        return output