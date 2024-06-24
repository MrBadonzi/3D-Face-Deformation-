from torch import nn as nn


class MLP_verts(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 32)
        self.lin6 = nn.Linear(32, 3)
        self.act = nn.LeakyReLU()

        self.conditioner = nn.Linear(4, 32)

    def forward(self, verts, exp):

        x = self.lin1(verts)
        x = self.act(x)

        y = self.conditioner(exp)
        x += y
        x = self.act(x)

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
