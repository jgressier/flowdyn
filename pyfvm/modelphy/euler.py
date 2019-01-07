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
        # Loop over all cells/control volumes
        pdata = []
        for i in range(self.neq+1):
            pdata.append(np.zeros(len(qdata[i]))) #test use zeros instead

        for c in range(len(qdata[0])):
            rho  = qdata[0][c] 
            rhou = qdata[1][c]
            rhoE = qdata[2][c]
            pdata[0][c]=rho                               # rho
            pdata[1][c]=rhou/rho                          # u
            pdata[2][c]=(rhoE-0.5*rhou*pdata[1][c])/rho   # e = E- u**2       
            pdata[3][c]=(self.gamma-1.0)*rho*pdata[2][c]  # p = (gamma-1)*rho*e       

        return pdata 

    def prim2cons(self, pdata): # qdata[ieq][cell] :
        # Loop over all cells/control volumes
        qdata = []
        for i in range(self.neq+1):
            qdata.append(np.zeros(len(pdata[i]))) #test use zeros instead

        # Loop over all cells/control volumes
        for c in range(len(pdata[0])):
            qdata[0][c] = pdata[0][c] 
            qdata[1][c] = pdata[0][c]*pdata[1][c] 
            qdata[2][c] = pdata[0][c]*(pdata[2][c]+pdata[1][c]**2) 
            qdata[3][c] = 0.0 # We don t need to acces to this data 
        return qdata

    def numflux(self, pdataL, pdataR): # HLLC Riemann solver ; pL[ieq][face]
        nflux = []
        for i in range(self.neq+1):
            nflux.append(np.zeros(len(pdataL[i]))) #test use zeros instead

        #  Loopdata over all faces
        for ifa in range(len(pdataL[0])):
            rhoL  = pdataL[0][ifa]
            uL    = pdataL[1][ifa]
            pL    = pdataL[3][ifa]
            rhoR  = pdataR[0][ifa]
            uR    = pdataR[1][ifa]
            pR    = pdataR[3][ifa]
            gamma = self.gamma
            # add ke to h
            # the enthalpy is assumed to include ke ...!
            HL = gamma/(gamma-1.0)*pL/rhoL + 0.5*uL**2
            HR = gamma/(gamma-1.0)*pR/rhoR + 0.5*uR**2

            # The HLLC Riemann solver
            
            #print ifa,gamma,pR,rhoR
            # raw_input('Press <ENTER> to continue')               #test
            cL   = math.sqrt(gamma*pL/rhoL)
            cR   = math.sqrt(gamma*pR/rhoR)

            unL  =  uL # We need to check the projection on the normals
            unR  =  uR # We need to check the projection on the normals
            uLuL = uL**2
            uRuR = uR**2

            # remove t       he   # essure term from the enthalpy, leaving an energy that has everything else...
            # sorry for using little "e" here - is is not just internal energy
            eL   = HL*rhoL-pL
            eR   = HR*rhoR-pR

            # Roe's averaging
            Rrho = math.sqrt(rhoR/rhoL)

            tmp    = 1.0/(1.0+Rrho);
            velRoe = tmp*(uL + uR*Rrho)
            uRoe   = tmp*(uL + uR*Rrho)
            hRoe   = tmp*(HL + HR*Rrho)

            gamPdivRho = tmp*( (gamma*pL/rhoL+0.5*(gamma-1.0)*uLuL) + (gamma*pR/rhoR+0.5*(gamma-1.0)*uRuR)*Rrho )
            cRoe  = math.sqrt(gamPdivRho - ((gamma+gamma)*0.5-1.0)*0.5*velRoe**2)

            # speed of sound at L and R
            sL = min(uRoe-cRoe, unL-cL)
            sR = max(uRoe+cRoe, unR+cR)

            # speed of contact surface
            sM = (pL-pR-rhoL*unL*(sL-unL)+rhoR*unR*(sR-unR))/(rhoR*(sR-unR)-rhoL*(sL-unL))

            # pressure at right and left (pR=pL) side of contact surface
            pStar = rhoR*(unR-sR)*(unR-sM)+pR

            if sM >= 0.0 :
                if sL > 0.0 :
                    Frho   = rhoL*unL
                    Frhou  = rhoL*uL*unL + pL
                    FrhoE  = (eL+pL)*unL
                else :
                    invSLmSs   = 1.0/(sL-sM)
                    sLmuL      = sL-unL
                    rhoSL      = rhoL*sLmuL*invSLmSs
                    eSL        = (sLmuL*eL-pL*unL+pStar*sM)*invSLmSs
                    rhouSL     = (rhoL*uL*sLmuL+(pStar-pL))*invSLmSs

                    Frho  = rhoSL*sM
                    Frhou = rhouSL*sM + pStar
                    FrhoE = (eSL+pStar)*sM
            else :
                if sR >= 0.0 :
                    invSRmSs = 1.0/(sR-sM)
                    sRmuR    = sR-unR
                    rhoSR    = rhoR*sRmuR*invSRmSs
                    eSR      = (sRmuR*eR-pR*unR+pStar*sM)*invSRmSs
                    rhouSR   = (rhoR*uR*sRmuR+(pStar-pR))*invSRmSs

                    Frho   = rhoSR*sM
                    Frhou  = rhouSR*sM + pStar
                    FrhoE  = (eSR+pStar)*sM
                else :
                    Frho   = rhoR*unR
                    Frhou  = rhoR*uR*unR + pR
                    FrhoE  = (eR+pR)*unR

            nflux[0][ifa] = Frho
            nflux[1][ifa] = Frhou
            nflux[2][ifa] = FrhoE
            nflux[3][ifa] = 0.0

        return nflux

    def timestep(self, data, dx, condition):
        "computation of timestep: data(=pdata) is not used, dx is an array of cell sizes, condition is the CFL number"
#        dt = CFL * dx / ( |u| + c )
        dt = np.zeros(len(dx)) #test use zeros instead
        for c in range(len(dx)):
            dt[c] = condition*dx[c]/ (data[1][c] + math.sqrt(self.gamma*data[3][c]/data[0][c]) )               
        return dt        


 
# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()

