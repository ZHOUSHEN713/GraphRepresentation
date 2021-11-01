import argparse
import os
import pickle

import networkx as nx
from Struc2VecWalk import generate_walks
from gensim.models import Word2Vec

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="brazil")
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--per_num', type=int, default=80)
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--depth', type=int, default=float('inf'))
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    # 航班信息，无向图
    graph = nx.read_edgelist(f"../data/flight/{args.dataset}_edges.txt",
                             create_using=nx.Graph(), nodetype=int, data=[("weight", float)])
    if not nx.is_weighted(graph):
        nx.set_edge_attributes(graph, values=1.0, name='weight')
    if not os.path.isfile(f"{args.dataset}_{args.per_num}_{args.length}.txt"):
        walks = generate_walks(graph, args.depth, args.per_num, args.length)
        pickle.dump(walks, open(f"{args.dataset}_{args.per_num}_{args.length}.txt", 'wb'))
    else:
        walks = pickle.load(open(f"{args.dataset}_{args.per_num}_{args.length}.txt", 'rb'))
    model = Word2Vec(sentences=walks, min_count=args.min_count, vector_size=args.emb_size, window=args.window_size,
                     workers=args.workers, sg=1, hs=1, epochs=args.epochs)
    model.save(f"{args.dataset}_{args.emb_size}_{args.epochs}.model")
