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
    Class model for euler equations

    attributes:
        _waves[5]

    """
    def __init__(self, gamma=1.4, source=None):
        base.model.__init__(self, name='euler', neq=3)
        self.islinear = 0
        self.gamma    = gamma
        self.source   = source
        self._vardict = { 'pressure': self.pressure, 'density': self.density,
                          'velocity': self.velocity, 'mach': self.mach }
        self._bcdict.update({'sym': self.bc_sym,
                         'insub': self.bc_insub,
                         'insup': self.bc_insup,
                         'outsub': self.bc_outsub,
                         'outsup': self.bc_outsup })
        
    def cons2prim(self, qdata): # qdata[ieq][cell] :
        """
        >>> model().cons2prim([[5.], [10.], [20.]])
        True
        """
        rho = qdata[0]
        u   = qdata[1]/qdata[0]
        p   = (self.gamma-1.0)*(qdata[2]-0.5*u*qdata[1])
        pdata = [ rho, u ,p ] 

        return pdata 

    def prim2cons(self, pdata): # qdata[ieq][cell] :
        """
        >>> model().prim2cons([[2.], [4.], [10.]]) == [[2.], [8.], [41.]]
        True
        """
        # Loop over all cells/control volumes
        qdata = []
        for i in range(self.neq):
            qdata.append(np.zeros(len(pdata[i]))) #test use zeros instead

        # Loop over all cells/control volumes
        for c in range(len(pdata[0])):
            qdata[0][c] = pdata[0][c] 
            qdata[1][c] = pdata[0][c]*pdata[1][c] 
            qdata[2][c] = pdata[2][c]/(self.gamma-1.) + .5*pdata[1][c]**2*pdata[0][c]
            #qdata[2][c] = pdata[0][c]*(pdata[2][c]+pdata[1][c]**2) 
        return qdata

    def density(self, qdata):
        return qdata[0].copy()

    def pressure(self, qdata):
        return (self.gamma-1.0)*(qdata[2]-0.5*qdata[1]**2/qdata[0])

    def velocity(self, qdata):
        return qdata[1]/qdata[0]

    def mach(self, qdata):
        return qdata[1]/np.sqrt(self.gamma*((self.gamma-1.0)*(qdata[2]-0.5*qdata[1]**2/qdata[0])))

    def numflux(self, pdataL, pdataR): # HLLC Riemann solver ; pL[ieq][face]

        gam  = self.gamma
        gam1 = gam-1.

        rhoL     = pdataL[0]
        uL = unL = pdataL[1]
        pL       = pdataL[2]
        rhoR     = pdataR[0]
        uR = unR = pdataR[1]
        pR       = pdataR[2]

        cL2   = gam*pL/rhoL
        cR2   = gam*pR/rhoR

        # the enthalpy is assumed to include ke ...!
        HL = cL2/gam1 + 0.5*uL**2
        HR = cR2/gam1 + 0.5*uR**2

        # The HLLC Riemann solver
                
        # sorry for using little "e" here - is is not just internal energy
        eL   = HL-pL/rhoL
        eR   = HR-pR/rhoR

        # Roe's averaging
        Rrho = np.sqrt(rhoR/rhoL)

        tmp    = 1.0/(1.0+Rrho);
        velRoe = tmp*(uL + uR*Rrho)
        uRoe   = tmp*(uL + uR*Rrho)
        hRoe   = tmp*(HL + HR*Rrho)

        gamPdivRho = tmp*( (cL2+0.5*gam1*uL*uL) + (cR2+0.5*gam1*uR*uR)*Rrho )
        cRoe  = np.sqrt(gamPdivRho - gam1*0.5*velRoe**2)

        # speed of sound at L and R
        sL = np.minimum(uRoe-cRoe, unL-np.sqrt(cL2))
        sR = np.maximum(uRoe+cRoe, unR+np.sqrt(cR2))

        # speed of contact surface
        sM = (pL-pR-rhoL*unL*(sL-unL)+rhoR*unR*(sR-unR))/(rhoR*(sR-unR)-rhoL*(sL-unL))

        # pressure at right and left (pR=pL) side of contact surface
        pStar = rhoR*(unR-sR)*(unR-sM)+pR

        # should not be computed if totally upwind
        SmoSSm = np.where(sM >= 0.,
                    sM/(sL-sM),
                    sM/(sR-sM))
        SmUoSSm = np.where(sM >= 0.,
                    (sL-unL)/(sL-sM),
                    (sR-unR)/(sR-sM))

        Frho = np.where(sM >= 0.,
                np.where(sL >= 0.,
                    rhoL*unL,
                    rhoL*sM*SmUoSSm),
                np.where(sR <= 0.,
                    rhoR*unR,
                    rhoR*sM*SmUoSSm))

        Frhou = np.where(sM >= 0.,
                np.where(sL >= 0.,
                    Frho*uL+pL,
                    Frho*uL + (pStar-pL)*SmoSSm + pStar),
                np.where(sR <= 0.,
                    Frho*uR+pR,
                    Frho*uR + (pStar-pR)*SmoSSm + pStar) )

        FrhoE = np.where(sM >= 0.,
                np.where(sL >= 0.,
                    rhoL*HL*unL,
                    Frho*eL + (pStar*sM-pL*unL)*SmoSSm + pStar*sM),
                np.where(sR <= 0.,
                    rhoR*HR*unR,
                    Frho*eR + (pStar*sM-pR*unR)*SmoSSm + pStar*sM))

        return [Frho, Frhou, FrhoE]

    def timestep(self, data, dx, condition):
        "computation of timestep: data(=pdata) is not used, dx is an array of cell sizes, condition is the CFL number"
        #        dt = CFL * dx / ( |u| + c )
        # dt = np.zeros(len(dx)) #test use zeros instead
        #dt = condition*dx/ (data[1] + np.sqrt(self.gamma*data[2]/data[0]) )
        dt = condition*dx *data[0]/ (
                np.abs(data[1]) + np.sqrt(
                self.gamma*(self.gamma-1.0)*(data[2]*data[0]-0.5*data[1]**2) ))
        return dt

    def bc_sym(self, dir, data, param):
        return

    def bc_insub(self, dir, data, param):
        return

    def bc_insup(self, dir, data, param):
        return param

    def bc_outsub(self, dir, data, param):
        return

    def bc_outsup(self, dir, data, param):
        return data
 
# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()

