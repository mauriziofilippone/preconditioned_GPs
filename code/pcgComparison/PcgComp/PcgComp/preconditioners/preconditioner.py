"""
Superclass for classes of Preconditioners.

"""
class Preconditioner(object):

    def __init__(self, name = ""):
        self.name = name
