import argparse
import logging
import networkx as nx
from Node2VecWalk import GenerateWalks
from gensim.models import Word2Vec


def GetLogger():
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s", datefmt="%a %b %d %H:%M:%S %Y")
    # set handler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    # add handler
    _logger.addHandler(sHandler)
    return _logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="wiki")
    parser.add_argument('--length', type=int, default=40)
    parser.add_argument('--per_num', type=int, default=30)
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--emb_size', type=int, default=2 ** 8)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--is_weighted', type=bool, default=False)
    parser.add_argument('--is_directed', type=bool, default=False)
    parser.add_argument('--p', type=float, default=2)
    parser.add_argument('--q', type=float, default=0.5)
    args = parser.parse_args()
    logger = GetLogger()
    # using nx.DiGraph create graph
    graph = nx.read_edgelist(f"../data/{args.dataset}/{args.dataset}_edges.txt",
                             create_using=nx.DiGraph(), nodetype=int, data=[("weight", float)])
    if not args.is_weighted:
        for e in graph.edges:
            graph[e[0]][e[1]]['weight'] = 1.0
    if not args.is_directed:
        graph = graph.to_undirected()
    walks = GenerateWalks(graph, args)
    logger.info("generate walks done!")
    # skip-gram --- hierarchical softmax
    model = Word2Vec(sentences=walks, min_count=args.min_count, vector_size=args.emb_size, window=args.window_size,
                     workers=args.workers, sg=1)
    logger.info("train Node2Vec done!")
