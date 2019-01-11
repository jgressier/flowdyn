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
    def __init__(self, gamma=1.4):
        base.model.__init__(self, name='euler', neq=3)
        self.islinear = 0
        self.gamma    = gamma
        
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

    def numflux(self, pdataL, pdataR): # HLLC Riemann solver ; pL[ieq][face]

        gam  = self.gamma
        gam1 = gam-1.

        rhoL  = pdataL[0]
        uL    = pdataL[1]
        pL    = pdataL[2]
        rhoR  = pdataR[0]
        uR    = pdataR[1]
        pR    = pdataR[2]

        # add ke to h
        # the enthalpy is assumed to include ke ...!
        HL = gam/gam1*pL/rhoL + 0.5*uL**2
        HR = gam/gam1*pR/rhoR + 0.5*uR**2

        # The HLLC Riemann solver
            
        cL   = np.sqrt(gam*pL/rhoL)
        cR   = np.sqrt(gam*pR/rhoR)

        unL  =  uL # We need to check the projection on the normals
        unR  =  uR # We need to check the projection on the normals
        uLuL = uL**2
        uRuR = uR**2

        # remove t       he   # essure term from the enthalpy, leaving an energy that has everything else...
        # sorry for using little "e" here - is is not just internal energy
        eL   = HL*rhoL-pL
        eR   = HR*rhoR-pR

        # Roe's averaging
        Rrho = np.sqrt(rhoR/rhoL)

        tmp    = 1.0/(1.0+Rrho);
        velRoe = tmp*(uL + uR*Rrho)
        uRoe   = tmp*(uL + uR*Rrho)
        hRoe   = tmp*(HL + HR*Rrho)

        gamPdivRho = tmp*( (gam*pL/rhoL+0.5*gam1*uLuL) + (gam*pR/rhoR+0.5*gam1*uRuR)*Rrho )
        cRoe  = np.sqrt(gamPdivRho - ((gam+gam)*0.5-1.0)*0.5*velRoe**2)

        # speed of sound at L and R
        sL = np.minimum(uRoe-cRoe, unL-cL)
        sR = np.maximum(uRoe+cRoe, unR+cR)

        # speed of contact surface
        sM = (pL-pR-rhoL*unL*(sL-unL)+rhoR*unR*(sR-unR))/(rhoR*(sR-unR)-rhoL*(sL-unL))

        # pressure at right and left (pR=pL) side of contact surface
        pStar = rhoR*(unR-sR)*(unR-sM)+pR

        Frho  = np.zeros(len(pdataL[0]))
        Frhou = np.zeros(len(pdataL[0]))
        FrhoE = np.zeros(len(pdataL[0]))

        for ifa in range(len(pdataL[0])):
            if sM[ifa] >= 0.0 :
                if sL[ifa] > 0.0 :
                    Frho[ifa]   = rhoL[ifa]*unL[ifa]
                    Frhou[ifa]  = rhoL[ifa]*uL[ifa]*unL[ifa] + pL[ifa]
                    FrhoE[ifa]  = (eL[ifa]+pL[ifa])*unL[ifa]
                else :
                    invSLmSs   = 1.0/(sL[ifa]-sM[ifa])
                    sLmuL      = sL[ifa]-unL[ifa]
                    rhoSL      = rhoL[ifa]*sLmuL*invSLmSs
                    eSL        = (sLmuL*eL[ifa]-pL[ifa]*unL[ifa]+pStar[ifa]*sM[ifa])*invSLmSs
                    rhouSL     = (rhoL[ifa]*uL[ifa]*sLmuL+(pStar[ifa]-pL[ifa]))*invSLmSs

                    Frho[ifa]  = rhoSL*sM[ifa]
                    Frhou[ifa] = rhouSL*sM[ifa] + pStar[ifa]
                    FrhoE[ifa] = (eSL+pStar[ifa])*sM[ifa]
            else :
                if sR >= 0.0 :
                    invSRmSs = 1.0/(sR[ifa]-sM[ifa])
                    sRmuR    = sR[ifa]-unR[ifa]
                    rhoSR    = rhoR[ifa]*sRmuR*invSRmSs
                    eSR      = (sRmuR*eR[ifa]-pR[ifa]*unR[ifa]+pStar[ifa]*sM[ifa])*invSRmSs
                    rhouSR   = (rhoR[ifa]*uR[ifa]*sRmuR+(pStar[ifa]-pR[ifa]))*invSRmSs

                    Frho[ifa]   = rhoSR*sM[ifa]
                    Frhou[ifa]  = rhouSR*sM[ifa] + pStar[ifa]
                    FrhoE[ifa]  = (eSR+pStar[ifa])*sM[ifa]
                else :
                    Frho[ifa]   = rhoR[ifa]*unR[ifa]
                    Frhou[ifa]  = rhoR[ifa]*uR[ifa]*unR[ifa] + pR[ifa]
                    FrhoE  = (eR[ifa]+pR[ifa])*unR[ifa]

        # for ifa in range(len(pdataL[0])):
        #     if sM >= 0.0 :
        #         if sL > 0.0 :
        #             Frho   = rhoL*unL
        #             Frhou  = rhoL*uL*unL + pL
        #             FrhoE  = (eL+pL)*unL
        #         else :
        #             invSLmSs   = 1.0/(sL-sM)
        #             sLmuL      = sL-unL
        #             rhoSL      = rhoL*sLmuL*invSLmSs
        #             eSL        = (sLmuL*eL-pL*unL+pStar*sM)*invSLmSs
        #             rhouSL     = (rhoL*uL*sLmuL+(pStar-pL))*invSLmSs

        #             Frho  = rhoSL*sM
        #             Frhou = rhouSL*sM + pStar
        #             FrhoE = (eSL+pStar)*sM
        #     else :
        #         if sR >= 0.0 :
        #             invSRmSs = 1.0/(sR-sM)
        #             sRmuR    = sR-unR
        #             rhoSR    = rhoR*sRmuR*invSRmSs
        #             eSR      = (sRmuR*eR-pR*unR+pStar*sM)*invSRmSs
        #             rhouSR   = (rhoR*uR*sRmuR+(pStar-pR))*invSRmSs

        #             Frho   = rhoSR*sM
        #             Frhou  = rhouSR*sM + pStar
        #             FrhoE  = (eSR+pStar)*sM
        #         else :
        #             Frho   = rhoR*unR
        #             Frhou  = rhoR*uR*unR + pR
        #             FrhoE  = (eR+pR)*unR

        return [Frho, Frhou, FrhoE]

    def timestep(self, data, dx, condition):
        "computation of timestep: data(=pdata) is not used, dx is an array of cell sizes, condition is the CFL number"
#        dt = CFL * dx / ( |u| + c )
        dt = np.zeros(len(dx)) #test use zeros instead
        for c in range(len(dx)):
            dt[c] = condition*dx[c]/ (data[1][c] + math.sqrt(self.gamma*data[2][c]/data[0][c]) )               
        return dt        


 
# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()

