# -*- coding: utf-8 -*-
"""
    The ``base`` module of modelphy library
    =========================
 
    Provides the Shallow water equations physical model.
    Initial conditions need to be pass in primitive variables, 
    i.e in [h_0, h_0*u_0].
    
    Avalable numerical flux : centered, rusanov, HLL.
    Availaible BC : inifite, symmetrical (not working). 
 
    :Example:
        model = shallowwater.model()
        w_init = [h0_vect, h0_vect*u0_vect]
        field0  = field.fdata(model, mesh, w_init)
 
    -------------------
    Author : Gaétan Foucart (06/01/2020) 

 """

import numpy as np
import math
import flowdyn.modelphy.base as mbase


# ===============================================================
# implementation of MODEL class

class model(mbase.model):
    """
    Class model for shallow water equations

    attributes:

    """
    def __init__(self, g=9.81, source=None):
        mbase.model.__init__(self, name='shallowwater', neq=2)
        self.islinear    = 0
        self.shape       = [1, 1]
        self.g           = g # gravity attraction
        self.source      = source
        
        self._vardict = { 'height': self.height, 'velocity': self.velocity, 'massflow': self.massflow}
        self._bcdict.update({'sym': self.bc_sym, 'infinite':self.bc_inf })
        self._numfluxdict = {'centered': self.numflux_centeredflux, 
                             'rusanov': self.numflux_rusanov, 'hll': self.numflux_hll }
        
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

    def height(self, qdata):
        return qdata[0].copy()
    
    def massflow(self,qdata):
        return qdata[1].copy()
    
    def velocity(self, qdata):
        return qdata[1]/qdata[0]

    def numflux(self, name, pdataL, pdataR, dir=None):
        if name==None: name='rusanov'
        return (self._numfluxdict[name])(pdataL, pdataR, dir)

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
        Fh = .5*( hL*uL + hR*uR ) - 0.5*cmax*(hR - hL)
        Fq = .5*( (hL*uL**2 + 0.5*g*hL**2) + (hR*uR**2 + 0.5*g*hR**2)) - 0.5*cmax*(hR*uR - hL*uL)

        return [Fh, Fq]

    def numflux_hll(self, pdataL, pdataR, dir=None): # HLL flux ; pL[ieq][face]
        g  = self.g
  
        hL = pdataL[0]
        uL = pdataL[1]
        hR = pdataR[0]
        uR = pdataR[1]
        
        # Eigen values definition        
        if (hL[:]>=0).all():
            lambda1L = uL + np.sqrt(g*hL)
            lambda2L = uL - np.sqrt(g*hL)
        else:
            print('WARNING : negative height value encountered !')
            lambda1L = uL + np.sqrt(g*abs(hL))
            lambda2L = uL - np.sqrt(g*abs(hL))
                                    
        if (hR[:]>=0).all():
            lambda1R = uR + np.sqrt(g*hR)
            lambda2R = uR - np.sqrt(g*hR)
        
        else:
            print('WARNING : negative height value encountered !')
            lambda1R = uR + np.sqrt(abs(g*hR))
            lambda2R = uR - np.sqrt(abs(g*hR))
        
        c1=np.zeros(len(hL))
        c2=np.zeros(len(hL))
        
        Fh = np.zeros(len(hL))
        Fu = np.zeros(len(hL))
        
        for i in range(len(hL)):
            c1[i] = min(lambda1L[i],lambda2L[i],lambda1R[i],lambda2R[i])
            c2[i] = max(lambda1L[i],lambda2L[i],lambda1R[i],lambda2R[i])
    
            if c1[i]>=0:
                Fh[i] = hL[i]*uL[i]
                Fu[i] = hL[i]*uL[i]**2 + 0.5*g*hL[i]**2
            
            elif c1[i]<0 and c2[i]>0:
                Fh_L = hL[i]*uL[i]
                Fh_R = hR[i]*uR[i]
                
                Fu_L = hL[i]*uL[i]**2 + 0.5*g*hL[i]**2
                Fu_R = hR[i]*uR[i]**2 + 0.5*g*hR[i]**2
                
                Fh[i] = (c2[i]*Fh_L-c1[i]*Fh_R)/(c2[i]-c1[i]) + (c1[i]*c2[i]/(c2[i]-c1[i]))*(hR[i]-hL[i])
                Fu[i] = (c2[i]*Fu_L-c1[i]*Fu_R)/(c2[i]-c1[i]) + (c1[i]*c2[i]/(c2[i]-c1[i]))*(hR[i]*uR[i]-hL[i]*uL[i])
                
            elif c2[i]<=0:
                Fh[i] = hR[i]*uR[i]
                Fu[i] = hR[i]*uR[i]**2 + 0.5*g*hR[i]**2

        return [Fh, Fu]
    
    def timestep(self, data, dx, condition):
        "computation of timestep: data(=pdata) is not used, dx is an array of cell sizes, condition is the CFL number"
        #        dt = CFL * dx / c where c is the highest eigen value velocity
        g = self.g
        c = abs(data[1]/data[0])+np.sqrt(g*data[0]) 
        dt = condition*dx / c 
        return dt

    def bc_sym(self, dir, data, param): # In primitive values here
        "symmetry boundary condition, for inviscid equations, it is equivalent to a wall, do not need user parameters"
        return [ data[0], -data[1]]

    def bc_inf(self, dir, data, param): # Simulate infinite plane : not sure...
        #zeros_h = np.zeros( np.shape(data[0]))
        #zeros_u = np.zeros( np.shape(data[1]))
        return [ data[0], data[1]] 

# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()

