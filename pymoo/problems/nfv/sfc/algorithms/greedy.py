
from ..network.network import Network
from ..network.requests import SFC, Request
import numpy as np
import functools
from copy import deepcopy
import sys
inf = sys.maxsize
EPS = 1e-3
def compare(a,b):
    if a[0] > b[0]:
        return 1
    else: 
        return -1
class Graph: 
    def __init__(self,network,request):
        self.request = request
        self.node_count = len(network.nodes)
        self.nodes = network.nodes
        self.capacity = np.ones((self.node_count,self.node_count))*-1
        self.tracing = []
        for link in network.links:
            src = int(link.u.name)
            trg = int(link.v.name)
            self.capacity[src][trg] = self.capacity[trg][src] = link.cap
    def dijkstra(self,source: int ,R: Request):
        d = np.ones(self.node_count) * -1
        trace = np.ones(self.node_count,dtype = int) * -1 
        depth = np.ones(self.node_count,dtype = int) * -1 
        d[source] = inf
        l = [(d[source],source)]
        while(len(l) !=0):
            l.sort(key=functools.cmp_to_key(compare))
            tp = l[0]
            l.pop(0)
            u = tp[1]
            if d[u] !=tp[0] :
                continue
            for idx in range(0,self.node_count):
                v = self.nodes[str(idx)]
                remain = min(d[u],self.capacity[u][int(v.name)] - R.bw)
                if(v.type == 1):
                    remain = min(remain,v.cap - R.mem)
                if remain > d[int(v.name)]:
                    assert remain >=0
                    # print(str(u) + " " + v.name + " " + str(remain))
                    d[int(v.name)] = remain
                    trace[int(v.name)] = u
                    depth[int(v.name)] = depth[u]+1
                    l.append((d[int(v.name)],int(v.name)))
        self.d = d
        self.trace = trace
        self.depth = depth
    def find_path(self,r: Request):
        source = int(r.ingress)
        sink = int(r.egress)
        nodes = deepcopy(self.nodes)
        capacity = deepcopy(self.capacity)
        path = []
        chosen_servers = []
        if nodes[str(source)].type == 1:
            if nodes[str(source)].cap < r.mem:
                return False
            nodes[str(source)].cap -= r.mem
        path.append(source)
        current_vertex = source
        for vnf in r.VNFs:
            self.dijkstra(current_vertex,r)
            best_next = best_value = best_depth = -1
            for next_vertex in range(self.node_count):
                if nodes[str(next_vertex)].has_VNF(vnf):
                    value = min(self.d[next_vertex],nodes[str(next_vertex)].cap - r.cpu) 
                    if value > best_value:
                        best_value = value
                        best_next = next_vertex
                        best_depth = self.depth[next_vertex]
                    elif value == best_value and value >=0 and self.depth[next_vertex] <best_depth:
                        best_next = next_vertex
                        best_depth = self.depth[next_vertex]
            if best_value < 0:
                return False
            u = best_next 
            while u != current_vertex:
                self.tracing.append(u)
                u = self.trace[u]
            nodes[str(best_next)].cap -= r.cpu
            while len(self.tracing) > 0:
                u = path[-1]  
                v = self.tracing[-1]
                capacity[u][v] -= r.bw
                capacity[v][u] -= r.bw
                if nodes[str(v)].type == 1:
                    nodes[str(v)].cap -= r.mem
                path.append(v)
                self.tracing.pop(-1)
            chosen_servers.append(len(path)-1)
            current_vertex = best_next
        self.dijkstra(current_vertex,r)
        if self.d[sink] < 0:
            return False
        u = sink
        while u != current_vertex:
            self.tracing.append(u)
            u = self.trace[u]
        while len(self.tracing) > 0:
            u = path[-1]  
            v = self.tracing[-1]
            capacity[u][v] -= r.bw
            capacity[v][u] -= r.bw
            if nodes[str(v)].type == 1:
                nodes[str(v)].cap -= r.mem
            path.append(v)
            self.tracing.pop(-1) 
        current_vertex = sink
        self.path = path
        self.chosen_servers = chosen_servers
        return True
    def path_to_sfc(self):
        sfc = SFC(self.request)
        sfc.route_nodes.append(self.nodes[self.request.ingress])
        for i in range(1,len(self.path)):
            u = self.nodes[str(self.path[i-1])]
            next_link = None
            for link in u.links:
                v = link.u if link.v == u else link.v
                if v.name == str(self.path[i]):
                    next_link = link
                    break
            sfc.route_nodes.append(self.nodes[str(self.path[i])])
            sfc.route_links.append(next_link)
        for vnf in self.chosen_servers:
            sfc.VNF_indices.append(vnf)
        return sfc
    def find_sfc(self):
        if self.find_path(self.request):
            return self.path_to_sfc()
        return None