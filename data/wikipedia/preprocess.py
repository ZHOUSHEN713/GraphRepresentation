import numpy as np
import scipy.io as scio


data = scio.loadmat("POS.mat")
matrix = np.array(data['network'].todense())
group = np.array(data['group'].todense())
edges, labels = [], []
for i in range(4777):
    for j in range(4777):
        if matrix[i][j] != 0:
            edges.append((i, j, matrix[i][j]))
    for k in range(40):
        if group[i][k] != 0:
            labels.append((i, k))
f = open("wikipedia_edges.txt", mode='w')
for e in edges:
    f.write(f"{e[0]} {e[1]} {e[2]}\n")
f.close()
f = open("wikipedia_labels.txt", mode='w')
for label in labels:
    f.write(f"{label[0]} {label[1]}\n")
f.close()
