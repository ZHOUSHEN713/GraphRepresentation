def preprocess_edges():
    edges = []
    with open("edges.csv", mode='r') as f:
        for e in f.readlines():
            x, y = e.split(',')
            edges.append([int(x), int(y)])
    f = open("BlogCatalog_edges.txt", mode='w')
    for e in edges:
        f.write(f"{e[0]-1} {e[1]-1}\n")
    f.close()


def preprocess_groups():
    labels = []
    with open("group-edges.csv", mode='r') as f:
        for line in f:
            x, y = line.split(',')
            labels.append([int(x), int(y)])
    f = open("BlogCatalog_labels.txt", mode='w')
    for label in labels:
        f.write(f"{label[0]-1} {label[1]-1}\n")


if __name__ == "__main__":
    preprocess_edges()
    preprocess_groups()
