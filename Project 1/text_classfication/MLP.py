import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features=29697,
                out_features=600,
                bias=True,
            ),
                nn.Tanh()
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(
                in_features=600,
                out_features=50,
            ),
            nn.Tanh()
        )

        self.classification = nn.Sequential(
            nn.Linear(
                in_features=50,
                out_features=10
            ),
            nn.ReLU()
        )

    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.classification(fc2)

        return output

