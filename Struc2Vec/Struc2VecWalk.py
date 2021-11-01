import math
import random

import numpy as np
import queue
from fastdtw import fastdtw
from collections import defaultdict
from tqdm import tqdm
from utils import create_alias_table, alias_sample


def get_rings_for_vertex(graph, root, max_depth):
    rings = {}
    visited = [0] * graph.number_of_nodes()
    cur_depth_num = 1  # 当前depth的个数
    next_depth_num = 0  # 下一层depth的个数
    depth = 0

    q = queue.Queue()
    q.put(root)
    cur_ring = {}
    while not q.empty():
        cur_depth_num -= 1
        cur = q.get()

        cur_ver_degree = len(graph[cur])  # 当前这个点的度数
        # 当前度数加1 (使用OPT2优化)
        cur_ring[cur_ver_degree] = cur_ring.get(cur_ver_degree, 0) + 1

        for v in graph[cur]:
            if not visited[v]:
                visited[v] = 1
                next_depth_num += 1
                q.put(v)

        if cur_depth_num == 0:
            ring = []
            for d, v in cur_ring.items():
                ring.append((d, v))
            ring.sort(key=lambda x: x[0])  # 按照度数排序
            cur_ring = {}
            rings[depth] = np.array(ring, dtype=np.int32)

            if depth == max_depth:
                break
            depth += 1

            cur_depth_num = next_depth_num
            next_depth_num = 0

    return rings


def get_rings_for_all_vertex(graph, max_depth):
    vertex_rings = {}  # 存储每个点所有环
    for v in tqdm(graph.nodes(), desc="get rings for all vertex"):
        vertex_rings[v] = get_rings_for_vertex(graph, v, max_depth)
    return vertex_rings


def create_vertex_degree(graph):
    degrees = defaultdict(list)
    S = set()
    for v in graph.nodes():
        d = len(graph[v])
        degrees[d].append(v)
        S.add(d)
    S = sorted(list(S))
    degree2idx = {b: a for a, b in enumerate(S)}
    idx2degree = {a: b for a, b in enumerate(S)}
    return degrees, degree2idx, idx2degree


def get_vertex_neighbors_by_binary_search(v_degree, degrees, degree2idx, idx2degree):
    vertex_ans = []
    visited = [0] * len(degrees)
    vertex_selected_num = 2 * math.log(len(degrees), 2)
    l = r = index = degree2idx[v_degree]
    while True:
        visited[index] = 1
        for c in degrees[idx2degree[index]]:
            vertex_ans.append(c)
            if len(vertex_ans) > vertex_selected_num:
                break
        # 使用双指针算法
        if l > 0 and visited[l]:
            l -= 1
        elif visited[l]:  # 处理边界点
            l = -1
        if r < len(degrees) - 1 and visited[r]:
            r += 1
        elif visited[r]:
            r = -1
        if l == -1 and r == -1:
            break
        elif l != -1 and r != -1:
            if v_degree - idx2degree[l] < idx2degree[r] - v_degree:
                index = l
            else:
                index = r
        elif l < 0:
            index = r
        else:
            index = l
    return vertex_ans


def get_all_vertex_neighbors(graph):
    degrees, degree2idx, idx2degree = create_vertex_degree(graph)
    vertex_need = {}
    for v in graph.nodes():
        d = len(graph[v])
        vertex_need[v] = get_vertex_neighbors_by_binary_search(d, degrees, degree2idx, idx2degree)
    return vertex_need


def cost_max(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * max(a[1], b[1])


def compute_vertex_distance(graph, max_depth):
    all_vertex_rings = get_rings_for_all_vertex(graph, max_depth)
    vertex_selected = get_all_vertex_neighbors(graph)
    distances = {}
    for v1, neighbors in tqdm(vertex_selected.items(), desc="compute_vertex_distance"):
        v1_layers = all_vertex_rings[v1]
        for v2 in neighbors:
            v2_layers = all_vertex_rings[v2]
            max_layer = min(len(v1_layers), len(v2_layers))
            distances[(v1, v2)] = {}
            for layer in range(max_layer):
                distance, _ = fastdtw(v1_layers[layer], v2_layers[layer], radius=1, dist=cost_max)
                distances[(v1, v2)][layer] = distance
    # convert to structural distance
    st = 1
    for vertex_pair, layer_distance in distances.items():
        layer_keys = sorted(layer_distance.keys())
        st = min(st, len(layer_keys))
        for layer in range(0, st):
            layer_keys.pop(0)
        for layer in layer_keys:
            layer_distance[layer] += layer_distance[layer - 1]

    return distances, vertex_selected


def convert_distance_shape(distance):
    layer_message, layer_distance = {}, {}
    for vertex_pair, layer_weight in distance.items():
        v1, v2 = vertex_pair
        for layer, weight in layer_weight.items():
            if layer not in layer_message:
                layer_message[layer] = {}
            if layer not in layer_distance:
                layer_distance[layer] = {}
            layer_distance[layer][(v1, v2)] = weight
            if v1 not in layer_message[layer]:
                layer_message[layer][v1] = []
            if v2 not in layer_message[layer]:
                layer_message[layer][v2] = []
            layer_message[layer][v1].append(v2)
            layer_message[layer][v2].append(v1)
    return layer_message, layer_distance


def build_graph(graph, max_depth):
    distance, vertex_selected = compute_vertex_distance(graph, max_depth)
    # distances: dict of dict
    layer_message, layer_distance = convert_distance_shape(distance)
    layer_alias, layer_accept, gamma = {}, {}, {}

    for layer in layer_message:
        v_neighbors = layer_message[layer]
        v_nbs_distances = layer_distance[layer]
        node_alias, node_accept = {}, {}
        node_weights = {}
        layer_weight, layer_cnt = 0.0, 0
        # 处理每一个点以及其邻居(k layer)
        for ver, neighbors in v_neighbors.items():
            sum_weight = 0.0
            weights = []
            for n in neighbors:
                if (ver, n) in v_nbs_distances:
                    w = v_nbs_distances[(ver, n)]
                else:
                    w = v_nbs_distances[(n, ver)]
                w = np.exp(-float(w))
                weights.append(w)
                sum_weight += w
            if sum_weight == 0.0:
                sum_weight = 1.0
            weights = [x / sum_weight for x in weights]
            node_alias[ver], node_accept[ver] = create_alias_table(weights)
            node_weights[ver] = weights
            layer_weight += sum(weights)
            layer_cnt += len(weights)
        layer_avg_weight = layer_weight / max(layer_cnt, 1)
        # 计算跨层转移的gamma值
        gamma[layer] = {}
        for ver, weights in node_weights.items():
            tt_cnt = 0
            for w in weights:
                if w > layer_avg_weight:
                    tt_cnt += 1
            gamma[layer][ver] = tt_cnt
        # 把每一层的转移存下来
        layer_alias[layer] = node_alias
        layer_accept[layer] = node_accept
    return layer_message, layer_alias, layer_accept, gamma


def bias_walk(start, length, layer_alias, layer_accept, layer_message, gamma, transfer_prob=0.3):
    path = [start]
    layer = 0
    while len(path) < length:
        # 在一层内进行转移
        cur = path[-1]
        if random.random() < transfer_prob:
            nbs = layer_message[layer][cur]
            idx = alias_sample(layer_accept[layer][cur], layer_alias[layer][cur])
            path.append(nbs[int(idx)])
        else:
            x = math.log(gamma[layer][cur] + math.e)
            # 向下转移
            if random.random() > (x / (x + 1)):
                if layer > 0:
                    layer -= 1
            # 向上转移
            else:
                if (layer + 1) in layer_message and cur in layer_message[layer + 1]:
                    layer += 1
    return path


def generate_walks(graph, max_depth, per_num, length):
    walks = []
    nodes = list(graph.nodes())
    layer_message, layer_alias, layer_accept, gamma = build_graph(graph, max_depth)
    for _ in tqdm(range(per_num), desc="generate walks"):
        random.shuffle(nodes)
        for ver in nodes:
            walks.append(bias_walk(ver, length, layer_alias, layer_accept, layer_message, gamma))
    return walks
