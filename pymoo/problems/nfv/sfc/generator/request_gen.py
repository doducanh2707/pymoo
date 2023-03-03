import glob
import os

import numpy as np
import random
from scipy.stats import poisson, expon

from ..generator.network_gen import file_to_network
from ..network.requests import Request
from ..network.network import Network


def generate_requests(network: Network, count=50, window=100, arrival_rate=0.05, mean_lifetime=500,
                      min_segments=1, max_segments=10, bandwidth=10, memory=10, cpu=10):
    # check invalid values
    assert not isinstance(cpu, list) or len(cpu) == 2
    assert not isinstance(memory, list) or len(memory) == 2
    assert not isinstance(bandwidth, list) or len(bandwidth) == 2

    requests = []

    timer = [0.0]
    cur = 0.0
    for _ in range(count):
        cur += random.expovariate(arrival_rate)
        timer.append(cur)

    lifetimes = expon.rvs(scale=mean_lifetime, size=count)
    # for cur in range(0, max_time, window):
    #     arrivals = poisson.rvs(mu=arrival_rate, k=window) + cur
    #     timer.extend(arrivals)
    # timer.sort()
    req_bw = np.random.gamma(bandwidth[0], bandwidth[1], count) \
        if isinstance(bandwidth, list) else np.full(count, bandwidth)
    req_mem = np.random.gamma(memory[0], memory[1], count) \
        if isinstance(memory, list) else np.full(count, memory)
    req_cpu = np.random.gamma(cpu[0], cpu[1], count) \
        if isinstance(cpu, list) else np.full(count, cpu)

    for arrival_time, lifetime, bw, mem, cpu1 in zip(timer, lifetimes, req_bw, req_mem, req_cpu):
        heads = [node.name for node in network.switch_nodes]
        ingress = random.choice(heads)
        egress = ingress
        while egress == ingress:
            egress = random.choice(heads)
        VNF_cnt = random.randint(min_segments, max_segments)
        VNFs = np.random.choice(np.arange(len(network.VNF_costs)), VNF_cnt)
        for i in range(1, len(VNFs)):
            while VNFs[i] == VNFs[i - 1]:
                VNFs[i] = np.random.choice(np.arange(len(network.VNF_costs)))

        req = Request(int(arrival_time), int(lifetime), ingress, egress, VNFs, int(bw), int(mem), int(cpu1))
        requests.append(req)

    return requests


def requests_to_file(requests, path):
    with open(path, 'w') as f:
        f.write(str(len(requests)) + '\n')
        for req in requests:
            f.write(' '.join([str(info) for info in [req.arrival, req.lifetime,
                                                     req.bw, req.mem, req.cpu, req.ingress, req.egress]]))
            f.write(' ' + ','.join([str(vnf) for vnf in req.VNFs]) + '\n')


def file_to_requests(path):
    with open(path) as f:
        R = int(f.readline())
        requests = []
        for _ in range(R):
            line = f.readline().split()
            arrival_time = float(line[0])
            lifetime = float(line[1])
            bandwidth = float(line[2])
            memory = float(line[3])
            cpu = float(line[4])
            ingress = line[5]
            egress = line[6]
            VNFs = [int(vnf) for vnf in line[7].split(',')]
            req = Request(arrival_time, lifetime, ingress, egress, VNFs, bandwidth, memory, cpu)
            requests.append(req)

        return requests


if __name__ == '__main__':
    net_folder = '../data/input'
    req_cnt = [10, 20, 30]
    for net_file in glob.glob(os.path.join(net_folder, '*network*')):
        print(net_file)
        network = file_to_network(net_file)
        bw = network.links[0].cap
        mem = network.switch_nodes[0].cap
        cpu = network.MDC_nodes[0].cap

        for req in req_cnt:
            requests = generate_requests(network, count=req, min_segments=1, max_segments=3,
                                         bandwidth=[bw/req, 2.0], memory=[mem/req, 2.0], cpu=[cpu/req, 2.0])
            requests_to_file(requests, net_file.replace('network', f'{req}requests'))
    # network = file_to_network(f'../data/input/{dataset}_network.txt')
    # requests = generate_requests(network, count=30, min_segments=2, max_segments=3)
    # requests_to_file(requests, f'../data/input/{dataset}_requests.txt')
    # requests = file_to_requests('./data/requests.txt')
