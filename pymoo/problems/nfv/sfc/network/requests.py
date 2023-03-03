
class Request:
    def __init__(self, arrival_time, lifetime, ingress, egress, VNFs: list, bandwidth=0, memory=0, cpu=0):
        # time
        self.arrival = arrival_time
        self.lifetime = lifetime

        # params
        self.ingress = ingress
        self.egress = egress
        self.VNFs = VNFs
        self.bw = bandwidth
        self.mem = memory
        self.cpu = cpu


class SFC:
    def __init__(self, request: Request):
        self.request = request
        self.route_nodes = []
        self.route_links = []
        self.VNF_indices = []

    # # add a node to the end of the route of this SFC
    # def next_switch(self, link, node):
    #     self.route_links.append(link)
    #     # next_node = link.next(self.route_nodes[-1])
    #     self.route_nodes.append(node)

    # # add a MDC node that contains the next requested VNF
    # def next_service(self, link):
    #     self.next_switch(link)
    #     assert self.route_nodes[-1].type == 2
    #     self.VNF_indices.append(len(self.route_links))

    # deploy this SFC and update the status of the network
    def deploy(self):
        # deployed_VNFs = []

        # cpu resource
        for i in range(len(self.VNF_indices)):
            node = self.route_nodes[self.VNF_indices[i]]
            node.use(self.request.cpu)
        # memory resources
        for node in self.route_nodes:
            if node.type == 1:
                node.use(self.request.mem)
        # bandwidth resource
        for link in self.route_links:
            link.use(self.request.bw)
        # # VNFs requested
        # for i, node_index in enumerate(self.VNF_indices):
        #     VNF_type = self.request.VNFs[i]
        #     node = self.route_nodes[node_index]
        #     if node.add_VNF(VNF_type):
        #         deployed_VNFs.append((node, VNF_type))

    # undo the deploy process
    def undeploy(self, deployed_VNFs=None):
        # cpu resource
        for i in range(len(self.VNF_indices)):
            node = self.route_nodes[self.VNF_indices[i]]
            node.use(-self.request.cpu)
        # memory resources
        for node in self.route_nodes:
            if node.type == 1:
                node.use(-self.request.mem)
        # bandwidth resource
        for link in self.route_links:
            link.use(-self.request.bw)
        # VNFs requested
        # for i, node_index in enumerate(self.VNF_indices):
        #     VNF_type = self.request.VNFs[i]
        #     node = self.route_nodes[node_index]
        #     node.use_VNF(VNF_type, -1)
        # if deployed_VNFs is not None:
        #     for node, VNF_type in deployed_VNFs:
        #         node.remove_VNF(VNF_type)

    # check if all VNF requirements are satisfied
    def check_vnf_constraint(self):
        # for i in range(len(self.VNF_indices)):
        #     node = self.route_nodes[self.VNF_indices[i]]
        #     k = self.request.VNFs[i]
        #     if node.type != 2 or node.violated_VNF(k) > 0:
        #         return False
        return True

    # check if all resource requirements are satisfied
    def check_capacity_constraints(self):
        # memory and cpu resources
        for node in self.route_nodes:
            if node.violated() > 0:
                return False
            # if isinstance(node, MDCNode) and (node.violated_cpu() > 0):
            #     return False
        # bandwidth resource
        for link in self.route_links:
            if link.violated() > 0:
                return False
        return True

    def check_feasibility(self):
        self.deploy()
        valid = (self.check_capacity_constraints() and self.check_vnf_constraint())
        self.undeploy()
        return valid

    def to_dict(self):
        return {
            'nodes': [node.name for node in self.route_nodes],
            'VNFs': self.VNF_indices
        }
