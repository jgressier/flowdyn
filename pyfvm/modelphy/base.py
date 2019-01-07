# -*- coding: utf-8 -*-
"""
    The ``base`` module of modelphy library
    =========================
 
    Provides ...
 
    :Example:
 
    >>> import hades.aero.Isentropic as Is
    >>> Is.TiTs_Mach(1.)
    1.2
    >>> Is.TiTs_Mach(2., gamma=1.6)
    2.2
 
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


 
# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()

