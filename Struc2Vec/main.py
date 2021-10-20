import argparse
import networkx as nx
from Struc2VecWalk import compute_vertex_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="wiki")
    args = parser.parse_args()
    # using nx.DiGraph create graph
    graph = nx.read_edgelist(f"../data/{args.dataset}/{args.dataset}_edges.txt",
                             create_using=nx.DiGraph(), nodetype=int, data=[("weight", float)])
    if not nx.is_weighted(graph):
        nx.set_edge_attributes(graph, values=1.0, name='weight')

    compute_vertex_distance(graph, 2)
