import argparse
import warnings
import torch
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim import Adam
from model import SDNE
from utils import get_logger
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
from TEST.classifier import train_test
from collections import defaultdict

warnings.filterwarnings('ignore')


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
                                     batch_size=args.batch_size)
        # 初始化模型参数
        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass

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
                loss = self.args.alpha * loss_1st + loss_2nd + self.args.reg * loss_reg
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                bar.set_description(f"Epoch {epoch}: Loss {round(loss.item() / batch_B.shape[0], 4)}")
            if epoch % 5 == 0:
                self.test()

    def get_embeddings(self):
        embeddings = {}
        with torch.no_grad():
            self.model.eval()
            vectors = self.model.predict(
                torch.FloatTensor(self.adj.todense()).to(self.args.device)).cpu().detach().numpy()
            for i in range(self.graph.number_of_nodes()):
                embeddings[str(i)] = vectors[i]
        return embeddings

    def test(self):
        embeddings = self.get_embeddings()
        data_x, data_y = [], []
        with open(f"../data/{self.args.dataset}/{self.args.dataset}_labels.txt", mode='r') as f:
            data = defaultdict(list)
            for line in f.readlines():
                x, y = line.split(" ")
                data[x].append(int(y))  # 注意label是不是从0开始
            for k, v in data.items():
                data_x.append(k)
                data_y.append(v)
        micros, macros = 0.0, 0.0
        for s in tqdm(range(20)):
            ans = train_test(data_x, data_y, embeddings, 0.2, s)
            # print(ans)
            micros += ans['micro']
            macros += ans['macro']
        print(micros / 20, macros / 20)


if __name__ == "__main__":
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="wiki")
    parser.add_argument('--alpha', type=float, default=1e-6)
    parser.add_argument('--beta', type=float, default=5)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--hidden_size1', type=int, default=256)
    parser.add_argument('--hidden_size2', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    # 读入的数据应该要提前经过映射
    graph = nx.read_edgelist(f"../data/{args.dataset}/{args.dataset}_edges.txt", create_using=nx.DiGraph(),
                             data=(("weight", float),), nodetype=int)
    if not nx.is_weighted(graph):
        nx.set_edge_attributes(graph, values=1.0, name='weight')
    if args.dataset in ["BlogCatalog", "wikipedia"]:
        graph = graph.to_undirected()
    model = SDNE(nodes=graph.number_of_nodes(), args=args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(graph, model, optimizer, args)
    trainer.train()
    trainer.test()
