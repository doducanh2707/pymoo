import os
import random
from collections import defaultdict

import numpy as np

from ..network.network import Network
from ..generator.graph_gen import file_to_edges2


def generate_network(graph, mdc_distribution, cpu, memory, bandwidth, vnf_cnt):
    # check invalid values
    assert not isinstance(cpu, list) or len(cpu) == 2
    assert not isinstance(memory, list) or len(memory) == 2
    assert not isinstance(bandwidth, list) or len(bandwidth) == 2

    network = Network([100] * vnf_cnt)

    # generate device costs and MDC nodes
    node_list, mdc_nodes = select_mdc_nodes(graph, mdc_distribution)
    for node in node_list:
        if node in mdc_nodes:
            if not isinstance(cpu, list):
                cap = cpu
            else:
                cap = random.uniform(cpu[0], cpu[1])
            # VNFs = np.random.choice(list(range(vnf_cnt)), int(vnf_cnt * np.random.uniform(0.1, 0.2)))
            network.add_MDC_node(node, cap)
        else:
            if not isinstance(memory, list):
                cap = memory
            else:
                cap = random.uniform(memory[0], memory[1])
            network.add_switch_node(node, cap)
    # extra_vnf_nodes = np.random.choice(network.MDC_nodes, vnf_cnt)
    # for i, node in enumerate(extra_vnf_nodes):
    #     node.add_VNF(i)

    distribute_vnf(network, mdc_distribution)

    # generate link bandwidth
    for u, v in graph:
        cap = bandwidth if not isinstance(bandwidth, list) \
            else random.uniform(bandwidth[0], bandwidth[1])
        network.add_link(u, v, cap)

    return network


def distribute_vnf(network, mdc_distribution):
    vnf_cnt = len(network.VNF_costs)
    if mdc_distribution == 'rural':
        extra_vnf_nodes = np.random.choice(network.MDC_nodes, vnf_cnt)
        for i, node in enumerate(extra_vnf_nodes):
            node.add_VNF(i)
        for node in network.MDC_nodes:
            if len(node.VNFs) == 0:
                node.add_VNF(np.random.randint(vnf_cnt))
    elif mdc_distribution == 'centers':
        for node in network.MDC_nodes:
            node.VNFs = set(range(vnf_cnt))
    else:
        for node in network.MDC_nodes:
            node.VNFs = set(np.random.choice(list(range(vnf_cnt)),
                                             int(vnf_cnt * np.random.uniform(0.1, 0.2))))
        extra_vnf_nodes = np.random.choice(network.MDC_nodes, vnf_cnt)
        for i, node in enumerate(extra_vnf_nodes):
            node.add_VNF(i)


def select_mdc_nodes(graph, mdc_distribution):
    # build adjacency list
    adj_list = defaultdict(set)
    for a, b in graph:
        adj_list[a].add(b)
        adj_list[b].add(a)
    node_list = list(adj_list.keys())
    node_list.sort(key=lambda node: -len(adj_list[node]))

    # select MDC nodes
    if mdc_distribution == 'uniform':
        mdc_nodes = random.sample(node_list, int(0.3 * len(node_list)))
    elif mdc_distribution == 'urban':
        mdc_nodes = node_list[:int(0.3 * len(node_list))]
    elif mdc_distribution == 'rural':
        mdc_nodes = node_list[int(0.7 * len(node_list)):]
    elif mdc_distribution == 'centers':
        mdc_nodes = node_list[:int(0.1 * len(node_list))]
    else:
        raise Exception('Invalid MDC distribution script!')

    return node_list, mdc_nodes


def network_to_file(network: Network, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(' '.join([str(cost) for cost in network.VNF_costs]) + '\n')
        f.write(str(len(network.nodes)) + '\n')
        for node in network.nodes.values():
            f.write(' '.join([node.name, str(node.cap), str(node.used)]))
            if node.type == 2:
                f.write(' [' + ','.join([str(vnf) for vnf in node.VNFs]) + ']')
            f.write('\n')

        f.write(str(len(network.links)) + '\n')
        for link in network.links:
            f.write(' '.join([link.u.name, link.v.name,
                              str(link.cap), str(link.used)]) + '\n')


def file_to_network(path: str):
    with open(path) as f:
        VNF_costs = [float(vnf) for vnf in f.readline().split()]
        network = Network(VNF_costs)
        N = int(f.readline())
        for _ in range(N):
            parts = f.readline().split()
            name = parts[0]
            cap = float(parts[1])
            used = float(parts[2])

            if len(parts) == 4:
                node_vnfs = []
                if len(parts[3]) > 2:
                    node_vnfs = [int(vnf) for vnf in parts[3][1:-1].split(',')]
                network.add_MDC_node(name, cap, used, node_vnfs)
            else:
                network.add_switch_node(name, cap, used)

        M = int(f.readline())
        for _ in range(M):
            parts = f.readline().split()
            network.add_link(parts[0], parts[1], float(parts[2]), float(parts[3]))

        return network


if __name__ == '__main__':
    # graph = generate_graph(10, 0.3)
    # graph_to_file(graph, './data/graph.txt')
    datasets = ['nsf', 'cogent', 'conus']
    distributions = ['uniform', 'urban', 'rural', 'centers']

    for dataset in datasets:
        graph = file_to_edges2(f'../data/topology/{dataset}_edges.txt')
        for dist in distributions:
            for i in range(5):
                if dataset == 'nsf':
                    network = generate_network(graph, dist, 100, 100, 100, 5)
                else:
                    network = generate_network(graph, dist, 100, 100, 100, 10)
                network_to_file(network, f'../data/input/{dataset}_{dist}_{i}_network.txt')
    # network = file_to_network(f'./data/{dataset}_nodes.csv')
    # network.visualize(pos_path=f'./data/{dataset}_nodes.csv')
