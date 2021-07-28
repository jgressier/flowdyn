# -*- coding: utf-8 -*-
"""
    The ``burgers`` module of modelphy library
    =========================

    Provides Burgers model

    :Example:

    >>> import flowdyn.modelphy.burgers as burg
    >>> model = burg.model()
    >>> print(model.neq, model.equation)
    1 burgers

    Available functions
    -------------------

 """

import numpy as np
import flowdyn.modelphy.base as base

# ===============================================================
# implementation of MODEL class

class model(base.model):
    """
    Class model for burgers equations
    Primitive and Conservative variables are the same

    attributes:

    """

    def __init__(self):
        base.model.__init__(self, name='burgers', neq=1)
        self.has_firstorder_terms = 1
        self.islinear = 0
        self.shape    = [1]

    def cons2prim(self, qdata): # conservative and primitive data are the same
        return qdata

    def prim2cons(self, pdata):  # conservative and primitive data are the same
        return pdata

    def numflux(self, name, pL, pR):
        """
        >>> model().numflux([[1.]], [[4.]]) == [.5]
        True
        >>> model().numflux([[-5.]],[[4.]]) == [8.]
        True
        """
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

