import random
from tqdm import tqdm


def RandomWalk(start, length, graph):
    """
    :param start: 单条轨迹开始点
    :param length: 轨迹最大长度
    :param graph: 图
    :return: 单条随机游走的轨迹
    """
    sequence = [start]
    while len(sequence) < length:
        cur = sequence[-1]
        neighbors = list(graph.neighbors(cur))
        if len(neighbors) > 0:
            sequence.append(random.choice(neighbors))
        else:
            break
    return sequence


def GenerateWalks(per_num, length, graph):
    walks = []
    nodes = list(graph.nodes())
    for _ in tqdm(range(per_num)):
        # 对于每一个node生成per_num条长度为length的序列
        random.shuffle(nodes)
        for ver in nodes:
            walks.append(RandomWalk(ver, length, graph))
    return walks
