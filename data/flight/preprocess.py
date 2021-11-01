with open("brazil_edges.txt", mode='r') as f:
    data = []
    for line in f.readlines():
        x, y = line.split(" ")
        cur = [int(x), int(y)]
        if cur not in data:
            data.append(cur)
    print(len(data))
