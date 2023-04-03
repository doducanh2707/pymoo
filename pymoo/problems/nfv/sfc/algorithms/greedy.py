import networkx as nx

from ..network.network import Network
from ..network.requests import SFC, Request

from .Node import *

EPS = 1e-3

class MultiLayerGraph:
    def __init__(self, network: Network):
        self.network = network
        self.G = nx.DiGraph()
        self.build()
    def build(self):
        for node in self.network.nodes.values():
            name = node.name
            self.G.add_node(name,
                            weight=(node.cap / (node.cap - node.used + 0.0001)
                                    if node.type == 1 else 0.0))
        for link in self.network.links:

            u = link.u.name
            v = link.v.name
            if u not in self.G.nodes or v not in self.G.nodes:
                continue
            self.G.add_edge(u, v,
                            weight=(link.cap / (link.cap - link.used+0.0001) + self.G.nodes[v]['weight']))
            self.G.add_edge(v, u,
                            weight=(link.cap / (link.cap - link.used+0.0001) + self.G.nodes[u]['weight']))


    def k_dijkstra(self,start,end, k=10):
        if k == 1:
            try:
                return [nx.shortest_path(self.G, start, end, weight='weight')]
            except:
                return []
        X = nx.shortest_simple_paths(self.G, start, end, weight='weight')
        paths = []
        for counter, path in enumerate(X):
            paths.append(path)
            if counter == k - 1:
                break
        return paths

    def path_pair(self, path):
        paths = []
        route_links = []
        paths.append(self.network.nodes[path[0]])
        for i in range(1, len(path)):
            name  = path[i]
            pname = path[i - 1]
            u = self.network.nodes[pname]
            next_link = None
            for link in u.links:
                v = link.u if link.v == u else link.v
                if v.name == name:
                    next_link = link
                    break
            paths.append(self.network.nodes[name])
            route_links.append(next_link)
        return paths,route_links

    def find_SFCs(self,start,end, k=10):
        paths = self.k_dijkstra(start,end,k)
        return [self.path_pair(path) if path is not None else None for path in paths]


class state: 
    def __init__(self,network,requests,policy):
        self.network = network
        self.requests = requests
        self.policy = policy
        self.next = None 
        self.curr = None
        self.path = None
        self.route_links = None
        self.cur_req = None
        self.H = None
        self.i = 0 
        self.path_k = None
    def update(self,path,route_links):    
        for node in path:
            if node.type == 1: 
                node.use(self.cur_req.mem)
        for link in route_links:
            link.use(self.cur_req.bw)

    def reset(self,path,route_links,deployed_VNFs):
        for node in path:
            if node.type == 1:
                if node.used ==0 and self.cur_req.mem !=0:
                    print("Wrong reset " +str((len(path))))
                node.use(-self.cur_req.mem)
        for node in deployed_VNFs:
            if node.type ==  1: 
                print("Wrong")
            node.use(-self.cur_req.cpu)
        for link in route_links:
            link.use(-self.cur_req.bw)
            
    def Routing(self,K: int = 10):
        paths = []
        for r in self.requests:
            self.i +=1
            src = r.ingress
            dest = r.egress
            self.cur_req = r
            self.curr = self.network.nodes[src]
            if self.curr.violated(r.mem) or self.network.nodes[dest].violated(r.mem):
                paths.append(None)
                continue
            path = [self.curr]
            route_links = []
            served = True
            deployed_VNFs = []
            VNF_indices = []
            for i in r.VNFs:
                vnf_node = []
                for v in self.network.MDC_nodes:
                    if i in v.VNFs:
                        vnf_node.append(v)
                self.path_k = self.getpaths(self.curr,vnf_node,K)
                if self.path_k is None:
                    print(r)
                n = self.select(vnf_node,self.path_k)
                if n == None:
                    served = False
                    break
                deployed_VNFs.append(n)
                if n.name == self.curr.name:
                    VNF_indices.append(len(path)-1)
                    self.curr = n
                    n.use(self.cur_req.cpu)
                    continue
                p,links = self.path_k[n.name]
                if p == None:
                    print(self.curr.name)
                    print(n.name)
                self.update(p,links)
                p.pop(0)
                path.extend(p)
                VNF_indices.append(len(path)-1)
                route_links.extend(links)
                self.curr = n 
                n.use(self.cur_req.cpu)
            p,links = self.getpath(self.curr,self.network.nodes[dest],K)
            if p == None or not served:
                if len(path) != 1:
                    self.reset(path,route_links,deployed_VNFs)
                paths.append(None)
                continue
            self.update(p,links)
            p.pop(0)
            path.extend(p)
            route_links.extend(links)
            sfc = SFC(r)
            sfc.route_nodes = path 
            sfc.route_links = route_links
            sfc.VNF_indices = VNF_indices
            paths.append(sfc)
        return paths
    def select(self,vnf_node,paths):
        next = None
        first = True
        pitority = float('-inf')
        for v in vnf_node:
            if v.violated(self.cur_req.cpu):
                continue
            self.path,self.route_links = paths[v.name]
            if self.path == None:
                continue
            if(first):
                next = v 
                self.next = v
                pitority = self.policy.GetOutput(self)
                first = False
                continue
            self.next = v
            tmp = self.policy.GetOutput(self)
            if(tmp > pitority):
                next = v
                pitority = tmp
        return next
    def isValid(self,path,links):
        for node in path:
            if node.type == 1 and node.violated(self.cur_req.mem):
                return False
        for link in links:
            if link.violated(self.cur_req.bw):
                return False
        return True

    def getpath(self,u,v,k):
        mlgraph = MultiLayerGraph(self.network)
        paths = mlgraph.find_SFCs(u.name,v.name,k)
        for path in paths: 
            if self.isValid(path[0],path[1]):
                return path
        return None,None
    def getpaths(self,u,vnf_node,k):
        mlgraph = MultiLayerGraph(self.network)

        rs = {}
        for v in vnf_node:
            paths = mlgraph.find_SFCs(u.name,v.name,k)
            for path in paths: 
                if self.isValid(path[0],path[1]):
                    rs[v.name] = path
                    break
            if v.name not in rs.keys():
                rs[v.name] = None,None
        return rs

    