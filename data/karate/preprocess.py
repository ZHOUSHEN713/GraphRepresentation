data = []
with open("karate_edges.txt", mode='r') as f:
    for line in f.readlines():
        x, y = line.split(" ")
        data.append([int(x), int(y)])

with open("karate_edges.txt", mode='w') as f:
    for t in data:
        f.write(f"{t[0]-1} {t[1]-1}\n")

