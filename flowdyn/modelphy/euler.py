# -*- coding: utf-8 -*-
"""
    The ``base`` module of modelphy library
    =========================
 
    Provides ...
 
    :Example:
 
    >>> import aerokit.aero.Isentropic as Is
    >>> Is.TiTs_Mach(1.)
    1.2
    >>> Is.TiTs_Mach(2., gamma=1.6)
    2.2
 
    Available functions
    -------------------
 
    Provides ...
 """

import numpy as np
import flowdyn.modelphy.base as mbase

# ===============================================================
def _vecmag(qdata):
    return np.sqrt(np.sum(qdata**2, axis=0))

def _vecsqrmag(qdata):
    return np.sum(qdata**2, axis=0)

def _sca_mult_vec(r, v):
    return r*v # direct multiplication thanks to shape (:)*(2,:)

def _vec_dot_vec(v1, v2):
    return np.einsum('ij,ij->j', v1, v2) 

def datavector(ux, uy, uz=None):
    return np.vstack([ux, uy]) if not uz else np.vstack([ux, uy, uz])

# ===============================================================
# implementation of MODEL class

class base(mbase.model):
    """
    Class model for euler equations

    attributes:

    """
    def __init__(self, gamma=1.4, source=None):
        mbase.model.__init__(self, name='euler', neq=3)
        self.islinear    = 0
        self.shape       = [1, 1, 1]
        self.gamma       = gamma
        self.source      = source
        self._vardict = { 'pressure': self.pressure, 'density': self.density,
                          'velocity': self.velocity, 'asound': self.asound, 'mach': self.mach, 'enthalpy': self.enthalpy,
                          'entropy': self.entropy, 'ptot': self.ptot, 'rttot': self.rttot, 'htot': self.htot }
        
    def cons2prim(self, qdata): # qdata[ieq][cell] :
        """
        Primitives variables are rho, u, p
        >>> model().cons2prim([[5.], [10.], [20.]])
        True
        """
        rho = qdata[0]
        u   = qdata[1]/qdata[0]
        p   = self.pressure(qdata)
        pdata = [ rho, u ,p ] 
        return pdata 

    def prim2cons(self, pdata): # qdata[ieq][cell] :
        """
        >>> model().prim2cons([[2.], [4.], [10.]]) == [[2.], [8.], [41.]]
        True
        """
        V2 = pdata[1]**2 if pdata[1].ndim==1 else _vecsqrmag(pdata[1])
        rhoe = pdata[2]/(self.gamma-1.) + .5*pdata[0]*V2
        return [ pdata[0], _sca_mult_vec(pdata[0], pdata[1]), rhoe ]

    def density(self, qdata):
        return qdata[0].copy()

    def pressure(self, qdata): # returns (gam-1)*( rho.et) - .5 * (rho.u)**2 / rho )
        return (self.gamma-1.0)*(qdata[2]-self.kinetic_energy(qdata))

    def velocity(self, qdata):  # returns (rho u)/rho, works for both scalar and vector
        return qdata[1]/qdata[0]

    def velocitymag(self, qdata):  # returns mag(rho u)/rho, depending if scalar or vector
        return np.abs(qdata[1])/qdata[0] if qdata[1].ndim==1 else _vecmag(qdata[1])/qdata[0]

    def kinetic_energy(self, qdata):  
        """volumic kinetic energy"""
        return .5*qdata[1]**2/qdata[0] if qdata[1].ndim==1 else .5*_vecsqrmag(qdata[1])/qdata[0]

    def asound(self, qdata):
        return np.sqrt(self.gamma*self.pressure(qdata)/qdata[0])

    def mach(self, qdata):
        return qdata[1]/np.sqrt(self.gamma*((self.gamma-1.0)*(qdata[0]*qdata[2]-0.5*qdata[1]**2)))

    def entropy(self, qdata): # S/r
        return np.log(self.pressure(qdata)/qdata[0]**self.gamma)/(self.gamma-1.)

    def enthalpy(self, qdata): 
        return (qdata[2]-0.5*qdata[1]**2/qdata[0])*self.gamma/qdata[0]

    def ptot(self, qdata):
        gm1 = self.gamma-1.
        return self.pressure(qdata)*(1.+.5*gm1*self.mach(qdata)**2)**(self.gamma/gm1)

    def rttot(self, qdata):
        ec = 0.5*qdata[1]**2/qdata[0]
        return ((qdata[2]-ec)*self.gamma + ec)/qdata[0]/self.gamma*(self.gamma-1.)

    def htot(self, qdata):
        ec = 0.5*qdata[1]**2/qdata[0]
        return ((qdata[2]-ec)*self.gamma + ec)/qdata[0]

    def numflux(self, name, pdataL, pdataR, dir=None):
        if name is None: name='hllc'
        return (self._numfluxdict[name])(pdataL, pdataR, dir)

    def numflux_centeredflux(self, pdataL, pdataR, dir=None): # centered flux ; pL[ieq][face]
        gam  = self.gamma
        gam1 = gam-1.

        rhoL     = pdataL[0]
        uL = unL = pdataL[1]
        pL       = pdataL[2]
        rhoR     = pdataR[0]
        uR = unR = pdataR[1]
        pR       = pdataR[2]

        cL2 = gam*pL/rhoL
        cR2 = gam*pR/rhoR
        HL  = cL2/gam1 + 0.5*uL**2
        HR  = cR2/gam1 + 0.5*uR**2

        # final flux
        Frho  = .5*( rhoL*unL + rhoR*unR )
        Frhou = .5*( (rhoL*unL**2 + pL) + (rhoR*unR**2 + pR))
        FrhoE = .5*( (rhoL*unL*HL) + (rhoR*unR*HR))

        return [Frho, Frhou, FrhoE]

    def numflux_centeredmassflow(self, pdataL, pdataR, dir=None): # centered flux ; pL[ieq][face]
        gam  = self.gamma
        gam1 = gam-1.

        rhoL     = pdataL[0]
        uL = unL = pdataL[1]
        pL       = pdataL[2]
        rhoR     = pdataR[0]
        uR = unR = pdataR[1]
        pR       = pdataR[2]

        cL2 = gam*pL/rhoL
        cR2 = gam*pR/rhoR
        HL  = cL2/gam1 + 0.5*uL**2
        HR  = cR2/gam1 + 0.5*uR**2

        # final flux
        Frho  = .5*( rhoL*unL + rhoR*unR )
        Frhou = .5*( Frho*(unL+unR) + pL + pR)
        FrhoE = .5*Frho*( HL + HR)

        return [Frho, Frhou, FrhoE]

    def numflux_hlle(self, pdataL, pdataR, dir=None): # HLLE Riemann solver ; pL[ieq][face]

        gam  = self.gamma
        gam1 = gam-1.

        rhoL     = pdataL[0]
        uL = unL = pdataL[1]
        pL       = pdataL[2]
        rhoR     = pdataR[0]
        uR = unR = pdataR[1]
        pR       = pdataR[2]

        cL2 = gam*pL/rhoL
        cR2 = gam*pR/rhoR
        HL  = cL2/gam1 + 0.5*uL**2
        HR  = cR2/gam1 + 0.5*uR**2

        # The HLLE Riemann solver
                
        # sorry for using little "e" here - is is not just internal energy
        eL   = HL-pL/rhoL
        eR   = HR-pR/rhoR

        # Roe's averaging
        Rrho = np.sqrt(rhoR/rhoL)
        #
        tmp    = 1.0/(1.0+Rrho);
        velRoe = tmp*(uL + uR*Rrho)
        uRoe   = tmp*(uL + uR*Rrho)
        hRoe   = tmp*(HL + HR*Rrho)

        gamPdivRho = tmp*( (cL2+0.5*gam1*uL*uL) + (cR2+0.5*gam1*uR*uR)*Rrho )
        cRoe  = np.sqrt(gamPdivRho - gam1*0.5*velRoe**2)

        # max HLL 2 waves "velocities"
        sL = np.minimum(0., np.minimum(uRoe-cRoe, unL-np.sqrt(cL2)))
        sR = np.maximum(0., np.maximum(uRoe+cRoe, unR+np.sqrt(cR2)))

        # final flux
        Frho  = (sR*rhoL*unL - sL*rhoR*unR + sL*sR*(rhoR-rhoL))/(sR-sL)
        Frhou = (sR*(rhoL*unL**2 + pL) - sL*(rhoR*unR**2 + pR) + sL*sR*(rhoR*unR-rhoL*unL))/(sR-sL)
        FrhoE = (sR*(rhoL*unL*HL) - sL*(rhoR*unR*HR) + sL*sR*(rhoR*eR-rhoL*eL))/(sR-sL)

        return [Frho, Frhou, FrhoE]

    def numflux_hllc(self, pdataL, pdataR, dir=None): # HLLC Riemann solver ; pL[ieq][face]

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
        Vmag = self.velocitymag(data)
        dt = condition*dx / ( Vmag  + np.sqrt(self.gamma*(self.gamma-1.0)*(data[2]/data[0]-0.5*Vmag**2) ))
        return dt


# ===============================================================
# implementation of euler 1D class

class euler1d(base):
    """
    Class model for 2D euler equations
    """
    #bcdict = mbase.methoddict()

    def __init__(self, gamma=1.4, source=None):
        base.__init__(self, gamma=gamma, source=source)
        self.shape       = [1, 1, 1]
        self._vardict.update({ 'massflow': self.massflow })
        # self._bcdict.update({'sym': self.bc_sym,
        #                  'insub': self.bc_insub,
        #                  'insup': self.bc_insup,
        #                  'outsub': self.bc_outsub,
        #                  'outsup': self.bc_outsup })
        self._numfluxdict = { 'hllc': self.numflux_hllc, 'hlle': self.numflux_hlle, 
                        'centered': self.numflux_centeredflux, 'centeredmassflow': self.numflux_centeredmassflow }

    def _derived_fromprim(self, pdata, dir):
        """
        returns rho, un, V, c2, H
        'dir' is ignored
        """
        c2 = self.gamma * pdata[2] / pdata[0]
        H  = c2/(self.gamma-1.) + .5*pdata[1]**2
        return pdata[0], pdata[1], pdata[1], c2, H

    def massflow(self,qdata): # for 1D model only
        return qdata[1].copy()

    @base._bcdict.register('sym')
    def bc_sym(self, dir, data, param):
        "symmetry boundary condition, for inviscid equations, it is equivalent to a wall, do not need user parameters"
        return [ data[0], -data[1], data[2] ]

    @base._bcdict.register('insub')
    def bc_insub(self, dir, data, param):
        g   = self.gamma
        gmu = g-1.
        p  = data[2]
        m2 = np.maximum(0., ((param['ptot']/p)**(gmu/g)-1.)*2./gmu)
        rh = param['ptot']/param['rttot']/(1.+.5*gmu*m2)**(1./gmu)
        return [ rh, -dir*np.sqrt(g*m2*p/rh), p ] 

    @base._bcdict.register('insup')
    def bc_insup(self, dir, data, param):
        # expected parameters are 'ptot', 'rttot' and 'p'
        g   = self.gamma
        gmu = g-1.
        p  = param['p']
        m2 = np.maximum(0., ((param['ptot']/param['p'])**(gmu/g)-1.)*2./gmu)
        rh = param['ptot']/param['rttot']/(1.+.5*gmu*m2)**(1./gmu)
        return [ rh, -dir*np.sqrt(g*m2*p/rh), p ] 

    @base._bcdict.register('outsub')
    def bc_outsub(self, dir, data, param):
        return [ data[0], data[1], param['p'] ] 

    @base._bcdict.register('outsup')
    def bc_outsup(self, dir, data, param):
        return data

class model(euler1d): # backward compatibility
    pass

# ===============================================================
# implementation of derived MODEL class

class nozzle(euler1d):
    """
    Class nozzle for euler equations with section term -1/A dA/dx (rho u, rho u2, rho u Ht)

    attributes:

    """
    def __init__(self, sectionlaw, gamma=1.4, source=None):
        nozsrc = [ self.src_mass, self.src_mom, self.src_energy ]
        allsrc = nozsrc # init all sources to nozzle sources
        if source: # additional sources ?
            for i,isrc in enumerate(source):
                if isrc:
                    allsrc[i] = lambda x,q: isrc(x,q)+nozsrc[i](x,q)
        euler1d.__init__(self, gamma=gamma, source=allsrc)
        self.sectionlaw = sectionlaw
 
    def initdisc(self, mesh):
        self.geomterm = 1./self.sectionlaw(mesh.centers())* \
                    (self.sectionlaw(mesh.xf[1:mesh.ncell+1])-self.sectionlaw(mesh.xf[0:mesh.ncell])) / \
                    (mesh.xf[1:mesh.ncell+1]-mesh.xf[0:mesh.ncell])
        return 

    def massflow(self,qdata):
        return qdata[1]*self.sectionlaw(mesh.centers())

    def src_mass(self, x, qdata):
        return -self.geomterm * qdata[1]

    def src_mom(self, x, qdata):
        return -self.geomterm * qdata[1]**2/qdata[0]

    def src_energy(self, x, qdata):
        ec = 0.5*qdata[1]**2/qdata[0]
        return -self.geomterm * qdata[1] * ((qdata[2]-ec)*self.gamma + ec)/qdata[0]

# ===============================================================
# implementation of euler 2D class

class euler2d(base):
    """
    Class model for 2D euler equations
    """
    def __init__(self, gamma=1.4, source=None):
        base.__init__(self, gamma=gamma, source=source)
        self.shape       = [1, 2, 1]
        self._vardict.update({ 'velocity_x': self.velocity_x, 'velocity_y': self.velocity_y,
                         })
        #self._bcdict.update({ #'sym': self.bc_sym,
                        #  'insub': self.bc_insub,
                        #  'insup': self.bc_insup,
                        #  'outsub': self.bc_outsub,
                        #  'outsup': self.bc_outsup 
        #                })
        self._numfluxdict = { #'hllc': self.numflux_hllc, 'hlle': self.numflux_hlle, 
                        'centered': self.numflux_centeredflux  }

    def _derived_fromprim(self, pdata, dir):
        """
        returns rho, un, V, c2, H
        'dir' is ignored
        """
        c2 = self.gamma * pdata[2] / pdata[0]
        un = _vec_dot_vec(pdata[1], dir) 
        H  = c2/(self.gamma-1.) + .5*_vecmag(pdata[1])
        return pdata[0], un, pdata[1], pdata[2], H, c2

    def velocity_x(self, qdata):  # returns (rho ux)/rho
        return qdata[1][0,:]/qdata[0]

    def velocity_y(self, qdata):  # returns (rho uy)/rho
        return qdata[1][1,:]/qdata[0]

    def mach(self, qdata):
        rhoUmag = _vecmag(qdata[1])
        return rhoUmag/np.sqrt(self.gamma*((self.gamma-1.0)*(qdata[0]*qdata[2]-0.5*rhoUmag**2)))

    def numflux_centeredflux(self, pdataL, pdataR, dir): # centered flux ; pL[ieq][face]
        rhoL, unL, VL, pL, HL, cL2 = self._derived_fromprim(pdataL, dir)
        rhoR, unR, VR, pR, HR, cR2 = self._derived_fromprim(pdataR, dir)  
        # final flux
        Frho  = .5*( rhoL*unL + rhoR*unR )
        Frhou = .5*( (rhoL*unL)*VL + pL*dir + (rhoR*unR)*VR + pR*dir)
        FrhoE = .5*( (rhoL*unL*HL) + (rhoR*unR*HR))

        return [Frho, Frhou, FrhoE]

# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()

