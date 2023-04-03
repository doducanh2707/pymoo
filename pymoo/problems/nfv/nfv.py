import numpy as np

import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem
from .sfc.algorithms.multi_layer import MultiLayerGraph
from .sfc.algorithms.greedy import state
from .sfc.algorithms.Node import *
from copy import deepcopy
class NFV(Problem):
    # def __init__(self,name,network, requests, bound: tuple = [0,1],n_pareto_points:int = 100):
    #     self.name =name
    #     self.network = network
    #     self.requests = requests
    #     x = np.linspace(0, 1, n_pareto_points)
    #     self.pareto_front = np.array([x, 1 - np.sqrt(x)]).T
    #     self.bound = bound
    #     self.dim = len(requests)
    def __init__(self, network,requests, **kwargs):
        n_var = len(requests)
        self.network = network
        self.requests = requests
        super().__init__(n_var=n_var, n_obj=2, xl=0, xu=1, vtype=float, **kwargs)
    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T
    def decode(self, x):
        return np.argsort(x)
    def _evaluate(self, x, out, *args, **kwargs):
        x = self.decode(x)
        f1 = np.zeros(len(x))
        f2 = np.zeros(len(x))
        for i in range(len(x)):
            a,b = self.calc(x[i])
            f1[i] = a
            f2[i] = b
        out["F"] = anp.column_stack([f1, f2])
        
    def calc(self, x):
        paths = []
        cnt = 0
        for idx in x:
            req = self.requests[idx]
            mlp = MultiLayerGraph(self.network,req)
            path = mlp.find_SFCs(k=1)
            if len(path) == 0:
                paths.append(None)
            else:
                path = path[0]
                path.deploy()
                if not self.network.validate():
                    paths.append(None)
                    path.undeploy()
                else:
                    cnt +=1
                    paths.append(path)
        f1 = cnt / len(self.requests)
        f2 = (self.network.max_used_bandwidth() +
                self.network.max_used_memory() + self.network.max_used_cpu())/3
        for path in paths:
            if path is not None:
                path.undeploy()     
        return 1-f1,f2

class NFV_GP(Problem):
    # def __init__(self,name,network, requests, bound: tuple = [0,1],n_pareto_points:int = 100):
    #     self.name =name
    #     self.network = network
    #     self.requests = requests
    #     x = np.linspace(0, 1, n_pareto_points)
    #     self.pareto_front = np.array([x, 1 - np.sqrt(x)]).T
    #     self.bound = bound
    #     self.dim = len(requests)
    def __init__(self, network,requests,name, **kwargs):
        n_var = len(requests)
        self.network = network
        self.requests = requests
        self.get_policy(name)
        super().__init__(n_var=n_var, n_obj=2, xl=0, xu=1, vtype=float, **kwargs)
    
    def get_policy(self,name):
        if 'nsf' in name:
            self.policy = DivNode()
            x = MinNode()
            y = MulNode()
            self.policy.AppendChild(x)
            self.policy.AppendChild(y)
            x.AppendChild(NHNode())
            x.AppendChild(BRNode())
            y.AppendChild(ERCNode(0.48779723481575676))
            y.AppendChild(NHNode())
        if 'conus' in name:
            self.policy = None
        if 'cogent' in name:
            self.policy = None
    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T
    def decode(self, x):
        return np.argsort(x)
    def _evaluate(self, x, out, *args, **kwargs):
        x = self.decode(x)
        f1 = np.zeros(len(x))
        f2 = np.zeros(len(x))
        for i in range(len(x)):
            a,b = self.calc(x[i])
            f1[i] = a
            f2[i] = b
        out["F"] = anp.column_stack([f1, f2])
    def calc(self, x):
        paths = []
        cnt = 0
        new_req = [self.requests[i] for i in x]
        network_tmp = deepcopy(self.network)
        s = state(network_tmp,new_req,self.policy)
        paths = s.Routing() 
        cnt = 0 
        f2 = (network_tmp.max_used_bandwidth() +
                network_tmp.max_used_memory() + network_tmp.max_used_cpu())/3
        for path in paths:
            if path is not None:
                path.undeploy() 
                cnt+=1
        f1 = cnt / len(self.requests)    
        return 1-f1,f2