import argparse
import os
import pickle
import logging
import networkx as nx
from RandomWalk import GenerateWalks
from gensim.models import Word2Vec


def getLogger():
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
    parser.add_argument('--dataset', default="BlogCatalog")
    parser.add_argument('--length', type=int, default=40)
    parser.add_argument('--per_num', type=int, default=80)
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--emb_size', type=int, default=2**7)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()
    logger = getLogger()
    # directed graph
    graph = nx.read_edgelist(f"../data/{args.dataset}/{args.dataset}_edges.txt", create_using=nx.DiGraph())
    if args.dataset in ["BlogCatalog"]:
        graph = graph.to_undirected()
    if not os.path.isfile(f"deepwalk_{args.per_num}_{args.length}.txt"):
        walks = GenerateWalks(args.per_num, args.length, graph)
        pickle.dump(walks, open(f"deepwalk_{args.per_num}_{args.length}.txt", 'wb'))
    else:
        walks = pickle.load(open(f"deepwalk_{args.per_num}_{args.length}.txt", 'rb'))

    logger.info("generate walks done!")
    # skip-gram --- hierarchical softmax
    model = Word2Vec(sentences=walks, min_count=args.min_count, vector_size=args.emb_size, window=args.window_size,
                     workers=args.workers, hs=1, sg=1, epochs=args.epochs)
    model.save(f"{args.dataset}_{args.emb_size}_{args.epochs}.model")
    logger.info("train DeepWalk done!")
