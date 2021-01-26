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

import numpy as np
import math
#import flowdyn.modelphy.base as base

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
    def __init__(self, name='not defined', neq=0):
        self.equation = name
        self.neq      = neq
        self.source   = None
        self.islinear = 0
        self.has_firstorder_terms  = 0
        self.has_secondorder_terms = 0
        self.has_source_terms      = 0
        self._vardict = { }
        self._bcdict  = { 'dirichlet': self.bc_dirichlet }

    def __repr__(self):
        print("model: ", self.equation)
        print("nb eq: ", self.neq)

    def list_bc(self):
        return ['per']+list(self._bcdict.keys())

    def list_var(self):
        return self._vardict.keys()

    def cons2prim(self):  # NEEDS definition by derived model
        print("cons2prim method not implemented")
    
    def prim2cons(self):  # NEEDS definition by derived model
        print("prim2cons method not implemented")
    
    def initdisc(self, mesh):
        return
    
    def numflux(self): # NEEDS definition by derived model
        pass
    
    def timestep(self, data, dx, condition):  # NEEDS definition by derived model
        pass

    def nameddata(self, name, data):
        return (self._vardict[name])(data)

    def namedBC(self, name, dir, data, param):
        return (self._bcdict[name])(dir, data, param)

    #------------------------------------
    # definition of boundary conditions with name bc_*
    
    def bc_dirichlet(self, dir, data, param):
        return param['prim']


 
# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()

