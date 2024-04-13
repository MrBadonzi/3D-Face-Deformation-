from torch import nn as nn


class MLP_verts(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 32)
        self.lin2 = nn.Linear(32, 128)
        self.lin3 = nn.Linear(128, 512)
        self.lin4 = nn.Linear(512, 128)
        self.lin5 = nn.Linear(128, 32)
        self.lin6 = nn.Linear(32, 3)
        self.act = nn.LeakyReLU()

    def forward(self, verts):

        x = self.lin1(verts)
        x = self.act(x)

        # Depth = 4
        x = self.lin2(x)
        x = self.act(x)

        x = self.lin3(x)
        x = self.act(x)

        x = self.lin4(x)
        x = self.act(x)

        x = self.lin5(x)
        x = self.act(x)

        x = self.lin6(x)

        return x
