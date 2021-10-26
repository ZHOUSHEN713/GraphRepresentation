import argparse
import networkx as nx
import torch
import random
from utils import get_logger, create_alias_table, alias_sample
from collections import defaultdict
from model import LINE
from torch.optim import SGD, Adam
from tqdm import tqdm
from TEST.classifier import train_test


class Trainer:
    def __init__(self, graph, model, optimizer, args):
        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.args = args

        self.nodes, self.edges = graph.nodes(), graph.edges()
        self.node_nums, self.edge_nums = graph.number_of_nodes(), graph.number_of_edges()
        self.node2idx, self.idx2node = {}, {}
        for ver in graph.nodes():
            index = len(self.node2idx)
            self.node2idx[ver] = index
            self.idx2node[index] = ver
        # 构建Alias采样表
        self.node_accept, self.node_alias = None, None
        self.edge_accept, self.edge_alias = None, None
        self.preprocess_probs()
        # 初始化模型参数
        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass

    def preprocess_probs(self):
        power = 0.75
        degree = defaultdict(int)
        for e in self.edges():
            if self.graph.is_directed():
                degree[self.node2idx[e[0]]] += self.graph[e[0]][e[1]]['weight']
            else:
                degree[self.node2idx[e[0]]] += self.graph[e[0]][e[1]]['weight']
                degree[self.node2idx[e[1]]] += self.graph[e[0]][e[1]]['weight']
        degree_sum = float(sum([degree[i] ** power for i in range(self.node_nums)]))
        normalized_probs = [degree[i] ** power / degree_sum for i in range(self.node_nums)]
        self.nodes = [v for v in range(self.node_nums)]
        self.node_accept, self.node_alias = create_alias_table(normalized_probs)

        weight_sum = float(sum([self.graph[e[0]][e[1]]['weight'] for e in self.edges()]))
        normalized_probs = [self.graph[e[0]][e[1]]['weight'] / weight_sum for e in self.edges()]
        self.edges = [(self.node2idx[x], self.node2idx[y]) for (x, y) in self.edges]
        self.edge_accept, self.edge_alias = create_alias_table(normalized_probs)

    def generate_batch(self):
        u_i, u_j, neg = [], [], []
        for _ in range(self.args.batch_size):
            cur_edge_index = alias_sample(self.edge_accept, self.edge_alias)
            cur_edge = self.edges[cur_edge_index]
            u_i.append(cur_edge[0])
            u_j.append(cur_edge[1])
            neg.append([])
            for _ in range(self.args.K):
                while True:
                    negative_ver = alias_sample(self.node_accept, self.node_alias)
                    if not self.graph.has_edge(self.idx2node[cur_edge[0]], self.idx2node[negative_ver]):
                        break
                neg[-1].append(negative_ver)
            # 无向图
            if not self.graph.is_directed():
                u_i.append(cur_edge[1])
                u_j.append(cur_edge[0])
                neg.append([])
                for _ in range(self.args.K):
                    while True:
                        negative_ver = alias_sample(self.node_accept, self.node_alias)
                        if not self.graph.has_edge(self.idx2node[cur_edge[1]], self.idx2node[negative_ver]):
                            break
                    neg[-1].append(negative_ver)
        return u_i, u_j, neg

    def train(self):
        num = int(len(self.edges) / self.args.batch_size)
        for epoch in range(1, self.args.epochs + 1):
            bar = tqdm(range(num))
            for _ in bar:
                v_i, v_j, neg = self.generate_batch()
                v_i, v_j, neg = torch.LongTensor(v_i), torch.LongTensor(v_j), torch.LongTensor(neg)
                loss = self.model(v_i, v_j, neg)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                bar.set_description(f"Epoch {epoch}: Loss {loss.item()}")
            if epoch % 10 == 0:
                self.test()

    def get_embeddings(self):
        weight = self.model.vertex_embedding.weight.detach().numpy()
        embeddings = {}
        for i in range(self.node_nums):
            embeddings[self.idx2node[i]] = weight[i]
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
        for s in tqdm(random.sample(range(1, 10000), 10)):
            ans = train_test(data_x, data_y, embeddings, 0.5, s)
            micros += ans['micro']
            macros += ans['macro']
        print(micros / 10, macros / 10)


if __name__ == "__main__":
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="wikipedia")
    parser.add_argument('--K', type=int, default=5)  # 负采样个数
    parser.add_argument('--order', type=int, default=2)  # 无向图只能计算二阶相似度
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    graph = nx.read_edgelist(f"../data/{args.dataset}/{args.dataset}_edges.txt", create_using=nx.DiGraph(),
                             data=[("weight", float)])
    if not nx.is_weighted(graph):
        nx.set_edge_attributes(graph, values=1.0, name='weight')
    if args.dataset in ["BlogCatalog", "wikipedia"]:
        graph = graph.to_undirected()

    line = LINE(graph.number_of_nodes(), args)
    # optimizer = SGD(line.parameters(), lr=args.lr, momentum=0.9)
    optimizer = Adam(line.parameters(), lr=args.lr)

    trainer = Trainer(graph, line, optimizer, args)
    trainer.train()
