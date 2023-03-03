from collections import defaultdict
from string import ascii_uppercase
import random
import os

alphabet = list(ascii_uppercase)


def id_to_name(node_id: int):
    if node_id < 26:
        return alphabet[node_id]
    return alphabet[node_id//26-1] + alphabet[node_id%26]


def generate_graph(nodes, density):
    # randomly generate edges
    # adj_list = defaultdict(list)
    # edge_list = []
    # sum_acc = np.cumsum(np.array(range(nodes - 1, 0, -1))) - 1
    # edge_cnt = int(nodes * (nodes - 1) / 2)
    # edge_ids = random.sample(range(edge_cnt), int(density * edge_cnt))
    # for edge_id in edge_ids:
    #     a = np.searchsorted(sum_acc, edge_id)
    #     b = edge_id + 1 if a == 0 else edge_id - sum_acc[a-1] + a
    #     a, b = id_to_name(a), id_to_name(b)
    #     adj_list[a].append(b)
    #     adj_list[b].append(a)
    #     edge_list.append((a, b))

    # connect components
    edge_set = set()
    components = {}
    # visited = set()

    # def dfs(u, cid):
    #     visited.add(u)
    #     components[cid].append(u)
    #     for v in adj_list[u]:
    #         if v not in visited:
    #             dfs(v, cid)

    for i in range(nodes):
        components[i] = [i]
        # if node not in visited:
        #     dfs(node, len(components)-1)

    while len(components) > 1:
        com1, com2 = random.sample([*components.keys()], 2)
        a = random.choice(components[com1])
        b = random.choice(components[com2])
        # adj_list[a].append(b)
        # adj_list[b].append(a)
        edge_set.add(frozenset([a, b]))
        components[com1].extend(components[com2])
        components.pop(com2, None)

    full_edges = int(nodes * (nodes-1) / 2)
    for _ in range(len(edge_set), int(density * full_edges)):
        a = b = -1
        while a == b or frozenset([a, b]) in edge_set:
            a = random.randint(0, nodes-1)
            b = random.randint(0, nodes-1)
        edge_set.add(frozenset([a, b]))

    edge_list = []
    for edge in edge_set:
        a, b = edge
        edge_list.append((id_to_name(a), id_to_name(b)))

    return edge_list


def graph_to_file(edge_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(str(len(edge_list)) + '\n')
        for a, b in edge_list:
            f.write(a + ' ' + b + '\n')


def file_to_edges(path):
    with open(path) as f:
        m = int(f.readline())
        edges = []
        for _ in range(m):
            u, v = f.readline().split()
            edges.append([u, v])
    return edges


def file_to_edges2(path):
    with open(path) as f:
        edges = []
        line = f.readline()
        while line:
            if len(line) < 3:
                break
            u, v, _ = line.split()
            edges.append([u, v])
            line = f.readline()
    return edges


def file_to_adjacency_list(path):
    with open(path) as f:
        m = int(f.readline())
        adj_list = defaultdict(list)
        for _ in range(m):
            u, v = f.readline().split()
            adj_list[u].append(v)
            adj_list[v].append(u)
    return adj_list


# if __name__ == '__main__':
#     graph = generate_graph(20, 0.3)
#     graph_to_file(graph, './data/graph.txt')
