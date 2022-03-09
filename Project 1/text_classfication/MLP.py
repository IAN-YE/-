import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features=5888,
                out_features=500,
                bias=True,
            ),
            nn.Tanh()
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(
                in_features=500,
                out_features=50,
            ),
            nn.Tanh()
        )

        self.classification = nn.Sequential(
            nn.Linear(
                in_features=50,
                out_features=10
            ),
            nn.Tanh()
        )

    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.classification(fc2)

        return fc1, fc2, output

