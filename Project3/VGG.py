import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, dropout):
        super(VGG, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 4096),nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1000),nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        x = self.block_1(x) #[1,14,14]
        x = self.block_2(x) #[1,7,7]
        output = self.classifier(x.view(x.size(0), -1))

        return output

# x = torch.rand(size=(1,1,28,28))
# net = VGG(0.5)
# print(nn.ReLU(x))