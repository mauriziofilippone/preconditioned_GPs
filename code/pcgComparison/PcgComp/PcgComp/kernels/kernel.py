"""
Superclass for classes of Kernel functions.

"""
class Kernel(object):

	def __init__(self, name = ""):
		self.name = name

	"""
	Computation of Kernel matrix for the given inputs - Noise excluded
	"""
	def K(self, X1, X2):
		raise NotImplementedError

	"""
	Computation of scalar Kernel matrix - for grid inputs
	"""
	def K_scalar(self, X1, X2, original_dimensions):
		raise NotImplementedError