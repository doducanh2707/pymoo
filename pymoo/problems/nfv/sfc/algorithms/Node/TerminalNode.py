import numpy as np

from .BaseNode import Node

class NHNode(Node):
    def __init__(self):
        super(NHNode,self).__init__()

    def __repr__(self):
        return 'NH'

    def _GetHumanExpressionSpecificNode( self, args ):
        return  'NH'
    
    def GetOutput( self, X):
        tmp = 0
        for path in X.path_k:
            tmp = max(tmp,len(path))
        return len(X.path)/tmp
class BRNode(Node):
    def __init__(self):
        super(BRNode,self).__init__()

    def __repr__(self):
        return 'BR'

    def _GetHumanExpressionSpecificNode( self, args ):
        return  'BR'
    
    def GetOutput( self, X):
        tmp =  float('inf')
        for link in X.route_links:
            tmp = min(tmp,(link.cap-link.used)/link.cap)
        return tmp

class DNode(Node):
    def __init__(self):
        super(DNode,self).__init__()

    def __repr__(self):
        return 'Degree'

    def _GetHumanExpressionSpecificNode( self, args ):
        return  'Degree'
    def GetOutput(self,X):
        return None
class TBRNode(Node):
    def __init__(self):
        super(TBRNode,self).__init__()
    def __repr__(self):
        return 'TBR'

    def _GetHumanExpressionSpecificNode( self, args ):
        return  'TBR'
    def GetOutput(self,X):
        tmp = 0 
        for link in X.route_links:
            tmp += (link.cap-link.used)/link.cap
        return tmp
class CPUNode(Node):
    def __init__(self):
        super(CPUNode,self).__init__()
    def __repr__(self):
        return 'CPU'

    def _GetHumanExpressionSpecificNode( self, args ):
        return  'CPU'
    def GetOutput(self,X):
        return X.next.used/ X.next.cap
class ERCNode(Node):
    def __init__(self,value = None):
        super(ERCNode,self).__init__()
        if value is None:
            self.value = np.random.rand()
        else: 
            self.value = value
    def __repr__(self):
        return str(self.value) 

    def _GetHumanExpressionSpecificNode( self, args ):
        return  str(self.value)
    def GetOutput(self,X):
        return self.value