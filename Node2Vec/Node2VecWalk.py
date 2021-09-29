import random
import numpy as np


def create_alias_table(area_ratio):
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


def get_alias_edge(graph, t, v, args):
    unnormalized_prob = []
    for cur in sorted(graph.neighbors(v)):
        if graph.has_edge(cur, t):
            unnormalized_prob.append(graph[v][cur]['weight'])
        elif cur == t:
            unnormalized_prob.append(graph[v][cur]['weight'] / args.p)
        else:
            unnormalized_prob.append(graph[v][cur]['weight'] / args.q)
    prob_sum = float(sum(unnormalized_prob))
    normalized_prob = [float(x) / prob_sum for x in unnormalized_prob]
    return create_alias_table(normalized_prob)


def preprocess_transition_probs(graph, args):
    alias_nodes = {}
    for cur in graph.nodes():
        unnormalized_prob = [graph[cur][ne]['weight'] for ne in sorted(graph.neighbors(cur))]
        prob_sum = float(sum(unnormalized_prob))
        normalized_prob = [float(x) / prob_sum for x in unnormalized_prob]
        alias_nodes[cur] = create_alias_table(normalized_prob)
    alias_edges = {}
    if args.is_directed:
        for e in graph.edges():
            alias_edges[e] = get_alias_edge(graph, e[0], e[1], args)
    else:
        for e in graph.edges():
            alias_edges[e] = get_alias_edge(graph, e[0], e[1], args)
            alias_edges[(e[1], e[0])] = get_alias_edge(graph, e[1], e[0], args)
    return alias_nodes, alias_edges


def BiasWalk(graph, start, length, a, b):
    sequence = [start]
    while len(sequence) < length:
        cur = sequence[-1]
        neighbors = sorted(graph.neighbors(cur))
        if len(neighbors) == 0:
            break
        if len(sequence) == 1:
            next = neighbors[alias_sample(a[cur][0], a[cur][1])]
            sequence.append(next)
        elif len(sequence) >= 2:
            prev = sequence[-2]
            next = neighbors[alias_sample(b[(prev, cur)][0], b[(prev, cur)][1])]
            sequence.append(next)
        else:
            break
    return sequence


def GenerateWalks(graph, args):
    a, b = preprocess_transition_probs(graph, args)
    walks = []
    nodes = list(graph.nodes())
    for _ in range(args.per_num):
        random.shuffle(nodes)
        for ver in nodes:
            walks.append(BiasWalk(graph, ver, args.length, a, b))
    return walks

