# -*- coding: utf-8 -*-
"""
    The ``base`` module of modelphy library
    =========================
 
    Provides virtual class for all other model
 
    :Example:
 
    >>> model = modelbase.model(name='test', neq=1)
    >>> import pyfvm.modelphy.base as modelbase
    >>> print model.neq, model.equation
    1 test
 
    Available functions
    -------------------
 
    Provides ...
 """

import numpy as np
import math
#import pyfvm.modelphy.base as base

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
        self.islinear = 0
        self.has_firstorder_terms  = 0
        self.has_secondorder_terms = 0
        self.has_source_terms      = 0

    def __repr__(self):
        print "model: ", self.equation
        print "nb eq: ", self.neq
        
    def cons2prim(self):
        print "cons2prim method not implemented"
    
    def prim2cons(self):
        print "prim2cons method not implemented"
    
    def numflux(self):
        pass
    
    def timestep(self, data, dx, condition):
        pass

    def nameddata(self, name, data):
        #method = getattr(self, name)
        #return method()
        #print self._dict[name]
        return (self._dict[name])(data)


 
# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()

