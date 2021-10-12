import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim import Adam
from model import SDNE
from utils import get_logger
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp


class IndexDataset(Dataset):
    def __init__(self, nodes):
        self.data = np.arange(nodes)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


class Trainer:
    def __init__(self, graph, model, optimizer, args):
        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.adj, self.lap_mat = self.laplacian_matrix()
        self.dataloader = DataLoader(IndexDataset(self.graph.number_of_nodes()),
                                     shuffle=True, batch_size=args.batch_size)

    def laplacian_matrix(self):
        node_size = self.graph.number_of_nodes()
        val, row, col = [], [], []
        for e in self.graph.edges():
            row.append(e[0])
            col.append(e[1])
            val.append(self.graph[e[0]][e[1]]['weight'])
        if self.graph.is_directed():
            # 有向图
            A = sp.csr_matrix((val, (row, col)), shape=(node_size, node_size))
            A_ = sp.csr_matrix((val + val, (row + col, col + row)), shape=(node_size, node_size))
            D = sp.diags(A_.sum(axis=1).flatten().tolist()[0])
            L = D - A_
        else:
            # 无向图
            A = sp.csr_matrix((val + val, (row + col, col + row)), shape=(node_size, node_size))
            D = sp.diags(A.sum(axis=1).flatten().tolist()[0])
            L = D - A
        return A, L

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            bar = tqdm(self.dataloader)
            for batch in bar:
                batch_adj, batch_lap_mat = self.adj[batch, :].todense(), self.lap_mat[batch][:, batch].todense()
                batch_adj, batch_lap_mat = torch.FloatTensor(batch_adj), torch.FloatTensor(batch_lap_mat)
                batch_adj, batch_lap_mat = batch_adj.to(self.args.device), batch_lap_mat.to(self.args.device)
                batch_B = torch.ones_like(batch_adj).to(self.args.device)
                batch_B[batch_adj != 0] = self.args.beta

                loss_1st, loss_2nd = self.model(batch_adj, batch_lap_mat, batch_B)
                loss_reg = 0.0
                for name, param in self.model.named_parameters():
                    if "weight" in name:
                        loss_reg += torch.sum(param.data * param.data)
                loss = loss_1st + loss_2nd + self.args.gamma * 0.5 * loss_reg
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                bar.set_description(f"Epoch {epoch}: Loss {loss.item()}")




if __name__ == "__main__":
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="wiki")
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=1e-2)
    parser.add_argument('--hidden_size1', type=int, default=1000)
    parser.add_argument('--hidden_size2', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    # 读入的数据应该要提前经过映射
    graph = nx.read_edgelist(f"../data/{args.dataset}/{args.dataset}_edges.txt", create_using=nx.DiGraph(),
                             nodetype=int, data=(("weight", float),))
    if not nx.is_weighted(graph):
        nx.set_edge_attributes(graph, values=1.0, name='weight')

    model = SDNE(nodes=graph.number_of_nodes(), args=args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(graph, model, optimizer, args)
    trainer.train()
