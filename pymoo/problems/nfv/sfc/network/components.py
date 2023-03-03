class Node:
    def __hash__(self):
        return hash(self.name)

    def __init__(self, name, type, capacity, used=0.0, VNFs=None):
        self.name = name                # name Node
        self.type = type                # type Node (1:switch node; 2:MDC node)
        self.cap = capacity             # capacity of Node (type=1: memory_capacity, type=2: cpu_capacity) 
        self.used = used  
        self.links = []            

        if VNFs is None:               # VNF exist in Node (possible with type=2) example [1,2,3]
            self.VNFs = set()
        else:
            self.VNFs = VNFs                

    # check how much memory or cpu used exceed the capacity
    # pass param > 0 if wanting to try without using
    def violated(self, res=0.0):
        return (self.used + res) > self.cap

    # check the existence of VNF in the current node
    def has_VNF(self, VNF):
        return self.type == 2 and VNF in self.VNFs

    # add VNF
    def add_VNF(self, VNF):
        if not self.has_VNF(VNF):
            self.VNFs.add(VNF)
            return True
        return False

    # add VNF
    def remove_VNF(self, VNF):
        if self.has_VNF(VNF):
            self.VNFs.remove(VNF)

    # add used memory or cpu (subtract if passing a negative value)
    def use(self, res):
        self.used += res


class Link:
    def __hash__(self):
        return hash((self.u.name, self.v.name))

    def __init__(self, u, v, bandwidth_capacity, used=0.0):
        self.u = u  # first node
        self.v = v  # second node
        self.cap = bandwidth_capacity
        self.used = used 

    # get the next node if a node go through this link
    def next(self, node):
        return self.u if node is self.v else \
            (self.v if node is self.u else None)

    # check violated bandwidth
    def violated(self, res=0.0):
        return (self.used + res) > self.cap

    # add used bandwidth
    def use(self, bandwidth):
        self.used += bandwidth
