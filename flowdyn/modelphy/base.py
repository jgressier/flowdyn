# -*- coding: utf-8 -*-
"""
    The ``base`` module of modelphy library
    =========================
 
    Provides virtual class for all other model
 
    :Example:
 
    >>> import flowdyn.modelphy.base as modelbase
    >>> model = modelbase.model(name='test', neq=1)
    >>> print(model.neq, model.equation)
    1 test
 
    Available functions
    -------------------
 
    Provides ...
 """

class methoddict():
    """decorator to register decorated method as specific and tagged in the class model
    """
    def __init__(self):
        self.dict = {}

    def register(self, name):
        def decorator(classmeth):
            self.dict[name] = classmeth
            return classmeth
        return decorator
        

# ===============================================================
# implementation of MODEL class

class model():
    """
    Class model (as virtual class)

    attributes:
        neq
        islinear            
        has_firstorder_terms 
        has_secondorder_terms
        has_source_terms     

    """
    _bcdict = methoddict()   # dict and associated decorator method to register BC

    def __init__(self, name='not defined', neq=0):
        self.equation = name
        self.neq      = neq
        self.source   = None
        self.islinear = 0
        self.has_firstorder_terms  = 0
        self.has_secondorder_terms = 0
        self.has_source_terms      = 0
        self._vardict = { }
        #self._bcdict  = { 'dirichlet': self.bc_dirichlet }

    def __repr__(self):
        print("model: ", self.equation)
        print("nb eq: ", self.neq)

    def list_bc(self):
        return ['per']+list(self._bcdict.dict.keys())

    def list_var(self):
        return self._vardict.keys()

    def cons2prim(self, qdata):  # NEEDS definition by derived model
        raise NameError("must be implemented in derived class")

    
    def prim2cons(self, pdata):  # NEEDS definition by derived model
        raise NameError("must be implemented in derived class")
    
    def initdisc(self, mesh):
        return
    
    def numflux(self, name, pL, pR): # NEEDS definition by derived model
        raise NameError("must be implemented in derived class")
    
    def timestep(self, data, dx, condition):  # NEEDS definition by derived model
        raise NameError("must be implemented in derived class")

    def nameddata(self, name, data):
        return (self._vardict[name])(data)

    def namedBC(self, name, dir, data, param):
        return (self._bcdict.dict[name])(self, dir, data, param)

    #------------------------------------
    # definition of boundary conditions with name bc_*
    
    @_bcdict.register('dirichlet')
    def bc_dirichlet(self, dir, data, param):
        return param['prim']

# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()