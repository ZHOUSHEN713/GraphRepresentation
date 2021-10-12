import argparse
import networkx as nx
import torch

from utils import get_logger, create_alias_table, alias_sample
from collections import defaultdict
from model import LINE
from torch.optim import SGD
from tqdm import tqdm


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
            degree[self.node2idx[e[0]]] += self.graph[e[0]][e[1]]['weight']
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


if __name__ == "__main__":
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="wiki")
    parser.add_argument('--is_weighted', type=bool, default=False)
    parser.add_argument('--is_directed', type=bool, default=True)
    parser.add_argument('--K', type=int, default=5)  # 负采样个数
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--emb_size', type=int, default=2 ** 8)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    graph = nx.read_edgelist(f"../data/{args.dataset}/{args.dataset}_edges.txt", create_using=nx.DiGraph(),
                             nodetype=int)
    if not args.is_weighted:
        for e in graph.edges:
            graph[e[0]][e[1]]['weight'] = 1.0
    if not args.is_directed:
        graph = graph.to_undirected()

    line = LINE(graph.number_of_nodes(), args)
    optimizer = SGD(line.parameters(), lr=args.lr, momentum=0.9)

    trainer = Trainer(graph, line, optimizer, args)
    trainer.train()
