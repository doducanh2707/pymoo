import json
from copy import deepcopy

from ..network.network import Network
from ..network.requests import SFC


class Solution:
    def __init__(self, network: Network, SFCs: list):
        self.network = network
        self.SFCs = SFCs

    def accept_ratio(self):
        return len(list(filter(None, self.SFCs))) / len(self.SFCs)

    def MLU(self):
        for sfc in self.SFCs:
            if sfc is not None:
                sfc.deploy()

        return max(self.network.max_used_bandwidth(),
                   self.network.max_used_memory(),
                   self.network.max_used_cpu())

    def evaluate(self, alpha=0.1):

        return (1-alpha) * self.accept_ratio() + alpha * self.MLU()

    def export(self, path: str):
        with open(path, 'w') as f:
            data = {
                'total': len(self.SFCs),
                'accepted': len(list(filter(None, self.SFCs))),
                'utilization': self.MLU(),
                'SFCs': {k: v.to_dict() for k, v in enumerate(self.SFCs) if v is not None}
            }
            json.dump(data, f)
