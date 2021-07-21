# -*- coding: utf-8 -*-
"""
    The ``convection`` module of modelphy library
    =========================

    Provides convection model

    :Example:

    >>> model(2.).numflux([10.],[50.])
    [20.0]

    Available functions
    -------------------

 """

import flowdyn.modelphy.base as base

# ===============================================================
# implementation of MODEL class

class model(base.model):
    """
    Class model for convection

    attributes:

    """
    _vardict = base.methoddict()

    def __init__(self, convcoef):
        base.model.__init__(self, name='convection', neq=1)
        self.has_firstorder_terms = 1
        self.convcoef = convcoef
        self.islinear = 1
        self.shape    = [1]
        self._vardict.merge(model._vardict)

    def cons2prim(self, qdata):
        return [ 1*d for d in qdata ]

    def prim2cons(self, pdata):
        return [ 1*d for d in pdata ]

    def numflux(self, name, pL, pR):
        """
        >>> model(2.).numflux([10.],[50.])
        [20.0]
        >>> model(-2.).numflux([10.],[50.])
        [-100.0]
        """
        return [ self.convcoef*(pL[0]+pR[0])/2.-abs(self.convcoef)*(pR[0]-pL[0])/2. ]

    def timestep(self, pdata, dx, condition):
        "computation of timestep: data is not used, dx is an array of cell sizes, condition is the CFL number"
        return condition*dx/abs(self.convcoef)

    @_vardict.register()
    def q(self, qdata):
        return qdata[0].copy()

# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()

