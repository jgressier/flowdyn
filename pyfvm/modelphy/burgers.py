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
    Class model for burgers equations

    attributes:
        _waves[5]

    """
    def __init__(self):
        base.model.__init__(self, name='burgers', neq=1)
        self.has_firstorder_terms = 1
        self.islinear = 0
                
    def cons2prim(self, qdata):
        return qdata
        
    def prim2cons(self, pdata):
        return pdata

    def numflux(self, pL, pR):
        nflux = []
        for i in range(self.neq):
            nflux.append(np.zeros(len(pL[i]))) #test use zeros instead
            for c in range(len(pL[i])):
                #1st order Upwind scheme
                vhalf = (pL[i][c]+pR[i][c])/2   
                if vhalf > 0:
                    nflux[i][c] = pL[i][c]**2/2
                elif vhalf < 0:
                    nflux[i][c] = pR[i][c]**2/2  
        return nflux
    
    def timestep(self, data, dx, condition):
        "computation of timestep: data is not used, dx is an array of cell sizes, condition is the CFL number"
#        dt = CFL * dx / |u|
        dt = np.zeros(len(dx)) #test use zeros instead
        for c in range(len(dx)):
            dt[c] = condition*dx[c]/ abs(data[0][c])
             
        return dt 
        

 
# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()
