import numpy as np

from .BaseNode import Node

class AddNode(Node):
	
	def __init__(self):
		super(AddNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '+'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' + ' + args[1] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return X0 + X1

class SubNode(Node):
	def __init__(self):
		super(SubNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '-'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' - ' + args[1] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return X0 - X1

class MulNode(Node):
	def __init__(self):
		super(MulNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '*'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' * ' + args[1] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return X0 * X1
	
class DivNode(Node):
	def __init__(self):
		super(DivNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '/'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' / ' + args[1] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		if X1 == 0:
			return 1
		return X0 /X1

class MaxNode(Node):
    def __init__(self):
        super(MaxNode,self).__init__()
        self.arity = 2

    def __repr__(self):
        return 'Max'

    def _GetHumanExpressionSpecificNode( self, args ):
        return 'Max( ' + args[0] + ',' + args[1] + ' )'

    def GetOutput( self, X ):
        X0 = self._children[0].GetOutput( X )
        X1 = self._children[1].GetOutput( X )
        return max(X0 , X1)
class MinNode(Node):
    def __init__(self):
        super(MinNode,self).__init__()
        self.arity = 2

    def __repr__(self):
        return 'Min'

    def _GetHumanExpressionSpecificNode( self, args ):
        return 'Min( ' + args[0] + ' ,' + args[1] + ' )'

    def GetOutput( self, X ):
        X0 = self._children[0].GetOutput( X )
        X1 = self._children[1].GetOutput( X )
        return min(X0 , X1)
class AnalyticQuotientNode(Node):
	def __init__(self):
		super(AnalyticQuotientNode,self).__init__()
		self.arity = 2
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'aq'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' / sqrt( 1 + ' + args[1] + '**2 ) )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return X0 / np.sqrt( 1 + np.square(X1) )

class PowNode(Node):

	def __init__(self):
		super(PowNode,self).__init__()
		self.arity = 2
		self.is_not_arithmetic = True

	def __repr__(self):
		return '^'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( '+args[0]+'**( ' + args[0] + ' ))'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return np.power(X0, X1)

	
class ExpNode(Node):
	def __init__(self):
		super(ExpNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'exp'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'exp( ' + args[0] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.exp(X0)


class LogNode(Node):
	def __init__(self):
		super(LogNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'log'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'log( ' + args[0] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.log( np.abs(X0) + 1e-6 )


class SinNode(Node):
	def __init__(self):
		super(SinNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'sin'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'sin( ' + args[0] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.sin(X0)

class CosNode(Node):
	def __init__(self):
		super(CosNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'cos'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'cos( ' + args[0] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.cos(X0)
