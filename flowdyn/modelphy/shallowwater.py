# -*- coding: utf-8 -*-
"""
    The ``shallowwater`` module of modelphy library
    =========================

    Provides the shallow water equations physical model

    Initial conditions need to be pass in primitive variables,
    i.e in [h_0, h_0*u_0].

    Available numerical flux : centered, rusanov, HLL.
    Availaible BC : inifite, symmetrical (not working).

    :Example:
        model = shallowwater.shallowwater1d()
        w_init = [h0_vect, h0_vect*u0_vect]
        field0  = field.fdata(model, mesh, w_init)

    -------------------
    Author : Gaétan Foucart (06/01/2020)

 """

import numpy as np
import flowdyn.modelphy.base as base

# ===============================================================
# implementation of MODEL class

#class base(model):
#    pass

class shallowwater1d(base.model):
    """
    Class model for shallow water equations

    attributes:

    """
    _bcdict = base.methoddict('bc_')   # dict and associated decorator method to register BC
    _vardict = base.methoddict()
    _numfluxdict = base.methoddict('numflux_')

    def __init__(self, g=9.81, source=None):
        base.model.__init__(self, name='shallowwater', neq=2)
        self.islinear    = 0
        self.shape       = [1, 1]
        self.g           = g # gravity attraction
        self.source      = source
        self._bcdict.merge(shallowwater1d._bcdict)
        self._vardict.merge(shallowwater1d._vardict)
        self._numfluxdict.merge(shallowwater1d._numfluxdict)

    def cons2prim(self, qdata): # qdata[ieq][cell] :
        """
        >>> model().cons2prim([[5.], [10.], [20.]])
        True
        """
        h   = qdata[0]
        u   = qdata[1]/qdata[0]
        pdata = [ h, u ]
        return pdata #p pdata : primitive variables

    def prim2cons(self, pdata): # qdata[ieq][cell] :
        """
        >>> model().prim2cons([[2.], [4.], [10.]]) == [[2.], [8.], [41.]]
        True
        """
        qdata = [ pdata[0], pdata[0]*pdata[1]]
        return qdata # Convervative variables

    @_vardict.register()
    def height(self, qdata):
        return qdata[0].copy()

    @_vardict.register()
    def massflow(self,qdata):
        return qdata[1].copy()

    @_vardict.register()
    def velocity(self, qdata):
        return qdata[1]/qdata[0]

    def numflux(self, name, pdataL, pdataR, dir=None):
        if name is None: name='rusanov'
        return (self._numfluxdict.dict[name])(self, pdataL, pdataR, dir)

    @_numfluxdict.register(name='centered')
    @_numfluxdict.register()
    def numflux_centeredflux(self, pdataL, pdataR, dir=None): # centered flux ; pL[ieq][face] : in primitive variables (h,u) !
        g  = self.g

        hL = pdataL[0]
        uL = pdataL[1]
        hR = pdataR[0]
        uR = pdataR[1]

        # final flux
        Fh = .5*( hL*uL + hR*uR )
        Fq = .5*( (hL*uL**2 + 0.5*g*hL**2) + (hR*uR**2 + 0.5*g*hR**2))

        return [Fh, Fq]

    @_numfluxdict.register()
    def numflux_rusanov(self, pdataL, pdataR, dir=None): # Rusanov flux ; pL[ieq][face]
        g  = self.g
        #
        hL = pdataL[0]
        uL = pdataL[1]
        hR = pdataR[0]
        uR = pdataR[1]
        cL = np.sqrt(g*hL)
        cR = np.sqrt(g*hR)
        # Eigen values definition
        cmax = np.maximum(abs(uL)+cL,abs(uR)+cR)
        # final Rusanov flux
        qL = hL*uL
        qR = hR*uR
        Fh = .5*( qL + qR ) - 0.5*cmax*(hR - hL)
        Fq = .5*( (qL*uL + 0.5*g*hL**2) + (qR*uR**2 + 0.5*g*hR**2)) - 0.5*cmax*(qR-qL)

        return [Fh, Fq]

    @_numfluxdict.register()
    def numflux_hll(self, pdataL, pdataR, dir=None): # HLL flux ; pL[ieq][face]
        g  = self.g

        hL = pdataL[0]
        uL = pdataL[1]
        hR = pdataR[0]
        uR = pdataR[1]

        cL = np.sqrt(g*hL)
        cR = np.sqrt(g*hR)
        sL = np.minimum(0., np.minimum(uL-cL, uR-cR))
        sR = np.maximum(0., np.maximum(uL+cL, uR+cR))
        kL = sR/(sR-sL)
        kR = -sL/(sR-sL)
        qL = hL*uL
        qR = hR*uR
        Fh = kL*qL + kR*qR - kR*sR*(hR-hL)
        Fu = kL*(qL*uL+.5*g*hL**2) + kR*(qR*uR+.5*g*hR**2) - kR*sR*(qR-qL)
        return [Fh, Fu]

    def timestep(self, data, dx, condition):
        "computation of timestep: data(=pdata) is not used, dx is an array of cell sizes, condition is the CFL number"
        #        dt = CFL * dx / c where c is the highest eigen value velocity
        g = self.g
        c = abs(data[1]/data[0])+np.sqrt(g*data[0])
        dt = condition*dx / c
        return dt

    @_bcdict.register()
    def bc_sym(self, dir, data, param): # In primitive values here
        "symmetry boundary condition, for inviscid equations, it is equivalent to a wall, do not need user parameters"
        return [ data[0], -data[1] ]

    @_bcdict.register()
    def bc_inf(self, dir, data, param): # Simulate infinite plane : not sure...
        #zeros_h = np.zeros( np.shape(data[0]))
        #zeros_u = np.zeros( np.shape(data[1]))
        return [ data[0], data[1]]

# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()

