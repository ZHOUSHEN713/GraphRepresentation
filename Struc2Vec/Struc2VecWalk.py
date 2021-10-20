import math

import numpy as np
import queue
from fastdtw import fastdtw
from collections import defaultdict


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

        cur_ver_degree = len(graph[cur])
        # 当前度数加1
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
    for v in graph.nodes():
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
    for v1, neighbors in vertex_selected.items():
        v1_layers = all_vertex_rings[v1]
        for v2 in neighbors:
            v2_layers = all_vertex_rings[v2]
            max_layer = min(len(v1_layers), len(v2_layers))
            distances[(v1, v2)] = {}
            for layer in range(max_layer):
                distance, _ = fastdtw(v2_layers[layer], v2_layers[layer], radius=1, dist=cost_max)
                distances[(v1, v2)][layer] = distance
    return distances


