import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDNE(nn.Module):
    def __init__(self, nodes, args):
        super(SDNE, self).__init__()
        self.args = args
        self.encode1 = nn.Linear(nodes, self.args.hidden_size1)
        self.encode2 = nn.Linear(self.args.hidden_size1, self.args.hidden_size2)

        self.decode1 = nn.Linear(self.args.hidden_size2, self.args.hidden_size1)
        self.decode2 = nn.Linear(self.args.hidden_size1, nodes)

    def forward(self, x, L_matrix, B_matrix):
        # x: (batch, nodes)
        y = F.relu(self.encode2(F.relu(self.encode1(x))))
        x_hat = F.relu(self.decode2(F.relu(self.decode1(y))))

        loss_1st = 2 * torch.trace(torch.matmul(torch.matmul(y.transpose(0, 1), L_matrix), y)) / x.shape[0]
        loss_2nd = torch.mean(torch.sum(torch.pow((x_hat - x) * B_matrix, 2), dim=1))

        return loss_1st, loss_2nd

    def predict(self, x):
        y = self.encode2(self.encode1(x))
        return y

