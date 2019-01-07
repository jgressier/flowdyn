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
import pyfvm.modelphy.base as base

# ===============================================================
# implementation of MODEL class

class model(base.model):
    """
    Class model for convection

    attributes:
        _waves[5]

    """
    def __init__(self, convcoef):
        base.model.__init__(self, name='convection', neq=1)
        self.has_firstorder_terms = 1
        self.convcoef = convcoef
        self.islinear = 1
        
    def cons2prim(self, qdata):
        return qdata
        
    def prim2cons(self, pdata):
        return pdata

    def numflux(self, pL, pR):
        return [ self.convcoef*(pL[0]+pR[0])/2-abs(self.convcoef)*(pR[0]-pL[0])/2 ]
    
    def timestep(self, pdata, dx, condition):
        "computation of timestep: data is not used, dx is an array of cell sizes, condition is the CFL number"
        return condition*dx/abs(self.convcoef)

 
# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()
