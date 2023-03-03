import networkx as nx

from ..network.network import Network
from ..network.requests import SFC, Request

EPS = 1e-3


class MultiLayerGraph:
    def __init__(self, network: Network, request: Request):
        self.network = network
        self.request = request
        self.G = nx.DiGraph()
        self.build()

    def targets(self):
        start = self.request.ingress + '_0'
        end = self.request.egress + '_' + str(len(self.request.VNFs))
        return start, end

    def build(self):
        # self.nodes['ingress'] = GraphNode('ingress', 0.0)
        # self.nodes['egress'] = GraphNode('egress', 0.0)
        # max_bw = self.network.max_used_bandwidth()
        # max_mem = self.network.max_used_memory()
        # max_cpu = self.network.max_used_cpu()

        for layer in range(len(self.request.VNFs) + 1):

            for node in self.network.nodes.values():
                name = node.name + '_' + str(layer)
                if node.cap > node.used:
                    self.G.add_node(name,
                                    weight=(node.cap / (node.cap - node.used)
                                            if node.type == 1 else 0.0))

            for link in self.network.links:
                if link.cap <= link.used:
                    continue

                u = link.u.name + '_' + str(layer)
                v = link.v.name + '_' + str(layer)
                if u not in self.G.nodes or v not in self.G.nodes:
                    continue
                self.G.add_edge(u, v,
                                weight=(link.cap / (link.cap - link.used) + self.G.nodes[v]['weight']))
                self.G.add_edge(v, u,
                                weight=(link.cap / (link.cap - link.used) + self.G.nodes[u]['weight']))

            requested_VNF = self.request.VNFs[layer - 1]
            for mdc_node in self.network.MDC_nodes:
                if requested_VNF in mdc_node.VNFs:
                    if layer > 0:
                        u = mdc_node.name + '_' + str(layer - 1)
                        v = mdc_node.name + '_' + str(layer)
                        if mdc_node.cap <= mdc_node.used:
                            continue
                        self.G.add_edge(u, v,
                                        weight=(mdc_node.cap / (mdc_node.cap - mdc_node.used)
                                                + self.G.nodes[v]['weight']))


    def k_dijkstra(self, k=10):
        start, end = self.targets()

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

    def path_to_sfc(self, path):
        sfc = SFC(self.request)
        sfc.route_nodes.append(self.network.nodes[self.request.ingress])

        for i in range(1, len(path)):
            name, layer = path[i].split('_')
            pname, player = path[i - 1].split('_')
            u = self.network.nodes[pname]
            next_link = None
            for link in u.links:
                v = link.u if link.v == u else link.v
                if v.name == name:
                    next_link = link
                    break
            if int(layer) == int(player) + 1:
                sfc.VNF_indices.append(len(sfc.route_nodes) - 1)
            else:
                sfc.route_nodes.append(self.network.nodes[name])
                sfc.route_links.append(next_link)
                # sfc.next_switch(next_link, self.network.nodes[name])

        return sfc

    def find_SFCs(self, k=10):
        paths = self.k_dijkstra(k)
        return [self.path_to_sfc(path) if path is not None else None for path in paths]

