from ..network.components import *
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


class Network:
    def __init__(self, VNF_costs=None):
        self._max_cpu = None
        self._max_memory = None
        self._max_bandwidth = None
        self.nodes = {}  # list of all nodes (switch + MDC)
        self.links = []  # list of connections
        self.VNF_costs = VNF_costs

    # list of switch nodes
    @property
    def switch_nodes(self):
        return [node for node in self.nodes.values() if node.type == 1]

    # add a switch node to network
    def add_switch_node(self, name, memory_capacity, memory_used=0.0):
        node = Node(name, 1, memory_capacity, memory_used)
        self.nodes[name] = node

    # list of MDC nodes
    @property
    def MDC_nodes(self):
        return [node for node in self.nodes.values() if node.type == 2]

    # add a MDC node to network
    def add_MDC_node(self, name, cpu_capacity, cpu_used=0.0, VNFs=None):
        node = Node(name, 2, cpu_capacity, cpu_used, VNFs)
        self.nodes[name] = node

    # add a connection
    def add_link(self, u, v, bandwidth_capacity, bandwidth_used=0.0):
        u = self.nodes[u]
        v = self.nodes[v]
        link = Link(u, v, bandwidth_capacity, bandwidth_used)
        self.links.append(link)
        link.u.links.append(link)
        link.v.links.append(link)

    def violated_bandwidth(self):
        return sum([link.violated() for link in self.links])

    def violated_memory(self):
        return sum([node.violated() for node in self.switch_nodes])

    def violated_cpu(self):
        return sum([node.violated() for node in self.MDC_nodes])

    @property
    def max_bandwidth(self):
        if self._max_bandwidth is None:
            self._max_bandwidth = max([link.cap for link in self.links])
        return self._max_bandwidth

    @property
    def max_memory(self):
        if self._max_memory is None:
            self._max_memory = max([node.cap for node in self.switch_nodes])
        return self._max_memory

    @property
    def max_cpu(self):
        if self._max_cpu is None:
            self._max_cpu = max([node.cap for node in self.MDC_nodes])
        return self._max_cpu

    def max_used_bandwidth(self):
        return max([link.used / link.cap for link in self.links])

    def max_used_memory(self):
        return max([node.used / node.cap for node in self.switch_nodes])

    def max_used_cpu(self):
        return max([node.used / node.cap for node in self.MDC_nodes])

    def validate(self):
        return self.violated_bandwidth() + self.violated_cpu() + self.violated_memory() <= 1e-9

    def to_graph(self):
        G = nx.Graph()
        for node in self.nodes.values():
            G.add_node(node.name, type=node.type, cap=node.cap, used=node.used, VNFs=node.VNFs)
        for link in self.links:
            G.add_edge(link.u.name, link.v.name, cap=link.cap, used=link.used)
        return G

    def visualize(self, pos_path=None, info=False, topo=False, out_path=None):
        G = self.to_graph()

        if pos_path is None:
            pos_nodes = nx.spring_layout(G)
        else:
            df = pd.read_csv(pos_path, skiprows=1)
            pos_nodes = df.to_numpy()
            pos_nodes[:, 1] = -pos_nodes[:, 1]
            pos_nodes = {str(i): coord for i, coord in enumerate(pos_nodes)}

        if topo:
            colors = ['blue' if G.nodes[node]['type'] == 1 else 'red' for node in G.nodes]
        else:
            colors = ['black']
        nx.draw(G, pos_nodes, node_size=30, node_color=colors, with_labels=True, font_size=5, font_color='white')

        if info:
            pos_attrs = {}
            for node, coords in pos_nodes.items():
                pos_attrs[node] = (coords[0], coords[1] + 10)

            node_attrs = {}
            for node in G.nodes:
                attrs = G.nodes[node]
                node_attrs[node] = str(attrs['used']) + '/' + str(attrs['cap'])
            nx.draw_networkx_labels(G, pos_attrs, labels=node_attrs, font_size=5)

            edge_attrs = {}
            for edge in G.edges:
                attrs = G.edges[edge]
                edge_attrs[edge] = str(attrs['used']) + '/' + str(attrs['cap'])
            nx.draw_networkx_edge_labels(G, pos_nodes, edge_labels=edge_attrs, font_size=5)
        if out_path is not None:
            plt.savefig(out_path)
        plt.show()
