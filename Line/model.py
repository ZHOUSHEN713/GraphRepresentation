import torch
import torch.nn as nn
import torch.nn.functional as F


class LINE(nn.Module):
    def __init__(self, nodes, args):
        super(LINE, self).__init__()
        self.args = args
        self.vertex_embedding = nn.Embedding(nodes, self.args.emb_size)
        if self.args.order == 2:
            self.context_embedding = nn.Embedding(nodes, self.args.emb_size)

    def forward(self, v_i, v_j, neg):
        v_i = self.vertex_embedding(v_i).to(self.args.device)  # (batch, dim)
        if self.args.order == 2:
            v_j = self.context_embedding(v_j).to(self.args.device)  # (batch, dim)
            neg = self.context_embedding(neg).to(self.args.device)  # (batch, K, dim)
        else:
            v_j = self.vertex_embedding(v_j).to(self.args.device)
            neg = self.vertex_embedding(neg).to(self.args.device)

        left = F.logsigmoid(torch.sum(v_i * v_j, dim=1))
        right = torch.matmul(neg, v_i.unsqueeze(-1)).squeeze(-1)  # (batch, K)
        right = torch.sum(F.logsigmoid(-right), dim=1)

        return -torch.mean(left + right)
