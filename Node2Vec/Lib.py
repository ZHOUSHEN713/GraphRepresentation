import networkx as nx
from node2vec import Node2Vec

dataset = "wiki"
graph = nx.read_edgelist(f"../data/{dataset}/{dataset}_edges.txt",
                             create_using=nx.DiGraph(), data=[("weight", float)])
if not nx.is_weighted(graph):
    nx.set_edge_attributes(graph, values=1.0, name='weight')
node2vec = Node2Vec(graph, p=0.25, q=4, workers=4)
model = node2vec.fit(min_count=0, vector_size=128, window=10, sg=1, epochs=1)
model.save(f"{dataset}_{0.25}_{4}_lib.model")
