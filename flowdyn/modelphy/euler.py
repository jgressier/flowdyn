# -*- coding: utf-8 -*-
"""
    The ``euler`` module of modelphy library
    =========================

    Provides Euler model

    :Example:

    >>> import aerokit.aero.Isentropic as Is
    >>> Is.TiTs_Mach(1.)
    1.2
    >>> Is.TiTs_Mach(2., gamma=1.6)
    2.2

    Available functions
    -------------------

 """

import numpy as np
#import math
#from numpy.lib.function_base import _angle_dispatcher
import flowdyn.modelphy.base as base
from flowdyn._data import *

# ===============================================================
# implementation of MODEL class

class euler(base.model):
    """
    Class model for euler equations

    attributes:

    """
    _bcdict = base.methoddict('bc_')   # dict and associated decorator method to register BC
    _vardict = base.methoddict()
    _numfluxdict = base.methoddict('numflux_')

    def __init__(self, gamma=1.4, source=None):
        base.model.__init__(self, name='euler', neq=3)
        self.islinear    = 0
        self.shape       = [1, 1, 1]
        self.gamma       = gamma
        self.source      = source
        self._bcdict.merge(euler._bcdict)
        self._vardict.merge(euler._vardict)
        self._numfluxdict.merge(euler._numfluxdict)

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

    @_vardict.register()
    def density(self, qdata):
        return qdata[0].copy()

    @_vardict.register()
    def pressure(self, qdata): # returns (gam-1)*( rho.et) - .5 * (rho.u)**2 / rho )
        return (self.gamma-1.0)*(qdata[2]-self.kinetic_energy(qdata))

    @_vardict.register()
    def velocity(self, qdata):  # returns (rho u)/rho, works for both scalar and vector
        return qdata[1]/qdata[0]

    @_vardict.register()
    def velocitymag(self, qdata):  # returns mag(rho u)/rho, depending if scalar or vector
        return np.abs(qdata[1])/qdata[0] if qdata[1].ndim==1 else _vecmag(qdata[1])/qdata[0]

    @_vardict.register(name="kinetic-energy")
    @_vardict.register()
    def kinetic_energy(self, qdata):
        """volumic kinetic energy"""
        return .5*qdata[1]**2/qdata[0] if qdata[1].ndim==1 else .5*_vecsqrmag(qdata[1])/qdata[0]

    @_vardict.register()
    def asound(self, qdata):
        return np.sqrt(self.gamma*self.pressure(qdata)/qdata[0])

    @_vardict.register()
    def mach(self, qdata):
        return qdata[1]/np.sqrt(self.gamma*((self.gamma-1.0)*(qdata[0]*qdata[2]-0.5*qdata[1]**2)))

    @_vardict.register()
    def entropy(self, qdata): # S/r
        return np.log(self.pressure(qdata)/qdata[0]**self.gamma)/(self.gamma-1.)

    @_vardict.register()
    def enthalpy(self, qdata):
        return (qdata[2]-0.5*qdata[1]**2/qdata[0])*self.gamma/qdata[0]

    @_vardict.register()
    def ptot(self, qdata):
        gm1 = self.gamma-1.
        return self.pressure(qdata)*(1.+.5*gm1*self.mach(qdata)**2)**(self.gamma/gm1)

    @_vardict.register()
    def rttot(self, qdata):
        ec = self.kinetic_energy(qdata)
        return ((qdata[2]-ec)*self.gamma + ec)/qdata[0]/self.gamma*(self.gamma-1.)

    @_vardict.register()
    def htot(self, qdata):
        ec = self.kinetic_energy(qdata)
        return ((qdata[2]-ec)*self.gamma + ec)/qdata[0]

    def _Roe_average(self, rhoL, uL, HL, rhoR, uR, HR):
        """returns Roe averaged rho, u, usound"""
        # Roe's averaging
        Rrho = np.sqrt(rhoR/rhoL)
        tmp    = 1.0/(1.0+Rrho)
        #velRoe = tmp*(uL + uR*Rrho)
        uRoe   = tmp*(uL + uR*Rrho)
        hRoe   = tmp*(HL + HR*Rrho)
        cRoe  = np.sqrt((hRoe - 0.5*uRoe**2)*(self.gamma-1.))
        return Rrho, uRoe, cRoe

    def numflux(self, name, pdataL, pdataR, dir=None):
        if name is None: name='hllc'
        return (self._numfluxdict.dict[name])(self, pdataL, pdataR, dir)

    @_numfluxdict.register(name='centered')
    @_numfluxdict.register()
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

    @_numfluxdict.register()
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

    @_numfluxdict.register()
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
        Rrho, uRoe, cRoe = self._Roe_average(rhoL, uL, HL, rhoR, uR, HR)
        # max HLL 2 waves "velocities"
        sL = np.minimum(0., np.minimum(uRoe-cRoe, unL-np.sqrt(cL2)))
        sR = np.maximum(0., np.maximum(uRoe+cRoe, unR+np.sqrt(cR2)))

        # final flux
        Frho  = (sR*rhoL*unL - sL*rhoR*unR + sL*sR*(rhoR-rhoL))/(sR-sL)
        Frhou = (sR*(rhoL*unL**2 + pL) - sL*(rhoR*unR**2 + pR) + sL*sR*(rhoR*unR-rhoL*unL))/(sR-sL)
        FrhoE = (sR*(rhoL*unL*HL) - sL*(rhoR*unR*HR) + sL*sR*(rhoR*eR-rhoL*eL))/(sR-sL)

        return [Frho, Frhou, FrhoE]

    @_numfluxdict.register()
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
        Rrho, uRoe, cRoe = self._Roe_average(rhoL, uL, HL, rhoR, uR, HR)

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
        "computation of timestep with conservative data"
        #        dt = CFL * dx / ( |u| + c )
        # dt = np.zeros(len(dx)) #test use zeros instead
        #dt = condition*dx/ (data[1] + np.sqrt(self.gamma*data[2]/data[0]) )
        Vmag = self.velocitymag(data)
        dt = condition*dx / ( Vmag  + np.sqrt(self.gamma*(self.gamma-1.0)*(data[2]/data[0]-0.5*Vmag**2) ))
        return dt

# ===============================================================
# implementation of euler 1D class

class euler1d(euler):
    """
    Class model for 1D euler equations
    """
    _bcdict = base.methoddict('bc_')
    _vardict = base.methoddict()
    _numfluxdict = base.methoddict('numflux_')

    def __init__(self, gamma=1.4, source=None):
        euler.__init__(self, gamma=gamma, source=source)
        self.shape       = [1, 1, 1]
        self._bcdict.merge(euler1d._bcdict)
        self._vardict.merge(euler1d._vardict)
        self._numfluxdict.merge(euler1d._numfluxdict)

    def _derived_fromprim(self, pdata, dir):
        """
        returns rho, un, V, c2, H
        'dir' is ignored
        """
        c2 = self.gamma * pdata[2] / pdata[0]
        H  = c2/(self.gamma-1.) + .5*pdata[1]**2
        return pdata[0], pdata[1], pdata[1], c2, H

    @_vardict.register()
    def massflow(self,qdata): # for 1D model only
        return qdata[1].copy()

    @_bcdict.register()
    def bc_sym(self, dir, data, param):
        "symmetry boundary condition, for inviscid equations, it is equivalent to a wall, do not need user parameters"
        return [ data[0], -data[1], data[2] ]

    @_bcdict.register()
    def bc_insub(self, dir, data, param):
        g   = self.gamma
        gmu = g-1.
        p  = data[2]
        m2 = np.maximum(0., ((param['ptot']/p)**(gmu/g)-1.)*2./gmu)
        rh = param['ptot']/param['rttot']/(1.+.5*gmu*m2)**(1./gmu)
        return [ rh, -dir*np.sqrt(g*m2*p/rh), p ]

    @_bcdict.register()
    def bc_insub_cbc(self, dir, data, param):
        g   = self.gamma
        gmu = g-1.
        p  = data[2]
        invcm = data[1]+dir*2*np.sqrt(g*p/data[0])/gmu
        adiscri = g*(g+1)/gmu*param['rttot']-.5*gmu*invcm**2
        a1 = (dir*invcm+np.sqrt(adiscri))*gmu/(g+1.)
        u1 = invcm-dir*2*a1/gmu
        f_m1sqr= 1.+.5*gmu*(u1/a1)**2
        rh1 = param['ptot']/param['rttot']/f_m1sqr**(1./gmu)
        p1 = param['ptot']/f_m1sqr**(g/gmu)
        return [ rh1, u1, p1 ]

    @_bcdict.register()
    def bc_insup(self, dir, data, param):
        # expected parameters are 'ptot', 'rttot' and 'p'
        g   = self.gamma
        gmu = g-1.
        p=param['p']
        m2 = np.maximum(0., ((param['ptot']/p)**(gmu/g)-1.)*2./gmu)
        rh = param['ptot']/param['rttot']/(1.+.5*gmu*m2)**(1./gmu)
        return [rh, -dir*np.sqrt(g*m2*p/rh), p]

    @_bcdict.register(name='outsub')
    @_bcdict.register()
    def bc_outsub_prim(self, dir, data, param):
        return [ data[0], data[1], param['p'] ]

    @_bcdict.register()
    def bc_outsub_qtot(self, dir, data, param):
        g   = self.gamma
        gmu = g-1.
        m2  = data[1]**2/(g*data[2]/data[0])
        fm2 = 1.+.5*gmu*m2
        rttot = data[2]/data[0]*fm2
        ptot  = data[2]*fm2**(g/gmu)
        # right (external) state
        p  = param['p']
        m2 = np.maximum(0., ((ptot/p)**(gmu/g)-1.)*2./gmu)
        rho = ptot/rttot/(1.+.5*gmu*m2)**(1./gmu)
        return [ rho, dir*np.sqrt(g*m2*p/rho), p ]

    @_bcdict.register()
    def bc_outsub_rh(self, dir, data, param):
        g   = self.gamma
        gmu = g-1.
        # pratio > Ms > Ws/a0
        p0 = data[2]
        pratio = param['p']/p0
        u0 = data[1]
        # relative shock Mach number Ms=(u0-Ws)/a0
        Ms2 = 1.+(pratio-1.)*(g+1.)/(2.*g)
        rhoratio = ((g+1.)*Ms2)/(2.+gmu*Ms2)
        Ws = u0 - dir*np.sqrt(g*p0/data[0]*Ms2)
        # right (external) state
        p1  = param['p']
        u1 = Ws + (u0-Ws)/rhoratio
        rho1 = data[0]*rhoratio
        return [ rho1, u1, p1 ]

    @_bcdict.register()
    def bc_outsub_nrcbc(self, dir, data, param):
        g   = self.gamma
        gmu = g-1.
        # 0 and 1 stand for internal/external
        p1 = param['p']
        # isentropic invariant p/rho**gam = cst
        rho1 = data[0]*(p1/data[2])**(1./g)
        # C- invariant (or C+ according to dir)
        a0 = np.sqrt(g*data[2]/data[0])
        a1 = np.sqrt(g*p1/rho1)
        u1 = data[1] + dir*2/gmu*(a1-a0)
        return [ rho1, u1, p1 ]

    @_bcdict.register()
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
    _bcdict = base.methoddict('bc_')
    _vardict = base.methoddict()
    _numfluxdict = base.methoddict('numflux_')

    def __init__(self, sectionlaw, gamma=1.4, source=None):
        nozsrc = [ self.src_mass, self.src_mom, self.src_energy ]
        allsrc = nozsrc # init all sources to nozzle sources
        if source: # additional sources ?
            for i,isrc in enumerate(source):
                if isrc:
                    allsrc[i] = lambda x,q: isrc(x,q)+nozsrc[i](x,q)
        euler1d.__init__(self, gamma=gamma, source=allsrc)
        self.sectionlaw = sectionlaw
        self._bcdict.merge(nozzle._bcdict)
        self._vardict.merge(nozzle._vardict)
        self._numfluxdict.merge(nozzle._numfluxdict)


    def initdisc(self, mesh):
        self._xc = mesh.centers()
        self.geomterm = 1./self.sectionlaw(self._xc)* \
                    (self.sectionlaw(mesh.xf[1:mesh.ncell+1])-self.sectionlaw(mesh.xf[0:mesh.ncell])) / \
                    (mesh.xf[1:mesh.ncell+1]-mesh.xf[0:mesh.ncell])
        return

    @_vardict.register()
    def massflow(self,qdata):
        return qdata[1]*self.sectionlaw(self._xc)

    def src_mass(self, x, qdata):
        return -self.geomterm * qdata[1]

    def src_mom(self, x, qdata):
        return -self.geomterm * qdata[1]**2/qdata[0]

    def src_energy(self, x, qdata):
        ec = 0.5*qdata[1]**2/qdata[0]
        return -self.geomterm * qdata[1] * ((qdata[2]-ec)*self.gamma + ec)/qdata[0]

# ===============================================================
# implementation of euler 2D class

class euler2d(euler):
    """
    Class model for 2D euler equations
    """
    _bcdict = base.methoddict('bc_')
    _vardict = base.methoddict()
    _numfluxdict = base.methoddict('numflux_')

    def __init__(self, gamma=1.4, source=None):
        euler.__init__(self, gamma=gamma, source=source)
        self.shape       = [1, 2, 1]
        self._bcdict.merge(euler2d._bcdict)
        self._vardict.merge(euler2d._vardict)
        self._numfluxdict.merge(euler2d._numfluxdict)

    def _derived_fromprim(self, pdata, dir):
        """
        returns rho, un, V, p, H, c2
        """
        c2 = self.gamma * pdata[2] / pdata[0]
        un = _vec_dot_vec(pdata[1], dir)
        H  = c2/(self.gamma-1.) + .5*_vecsqrmag(pdata[1])
        return pdata[0], un, pdata[1], pdata[2], H, c2

    @_vardict.register()
    def velocity_x(self, qdata):  # returns (rho ux)/rho
        return qdata[1][0,:]/qdata[0]

    @_vardict.register()
    def velocity_y(self, qdata):  # returns (rho uy)/rho
        return qdata[1][1,:]/qdata[0]

    @_vardict.register()
    def mach(self, qdata):
        rhoUmag = _vecmag(qdata[1])
        return rhoUmag/np.sqrt(self.gamma*((self.gamma-1.0)*(qdata[0]*qdata[2]-0.5*rhoUmag**2)))

    def _Roe_average(self, rhoL, unL, UL, HL, rhoR, unR, UR, HR):
        """returns Roe averaged rho, u, usound"""
        # Roe's averaging
        Rrho = np.sqrt(rhoR/rhoL)
        tmp    = 1.0/(1.0+Rrho)
        unRoe  = tmp*(unL + unR*Rrho)
        URoe   = tmp*(UL + UR*Rrho)
        hRoe   = tmp*(HL + HR*Rrho)
        cRoe  = np.sqrt((hRoe - 0.5*_vecsqrmag(URoe))*(self.gamma-1.))
        return Rrho, unRoe, cRoe

    @_numfluxdict.register(name='centered')
    @_numfluxdict.register()
    def numflux_centeredflux(self, pdataL, pdataR, dir): # centered flux ; pL[ieq][face]
        rhoL, unL, VL, pL, HL, cL2 = self._derived_fromprim(pdataL, dir)
        rhoR, unR, VR, pR, HR, cR2 = self._derived_fromprim(pdataR, dir)
        # final flux
        Frho  = .5*( rhoL*unL + rhoR*unR )
        Frhou = .5*( (rhoL*unL)*VL + pL*dir + (rhoR*unR)*VR + pR*dir)
        FrhoE = .5*( (rhoL*unL*HL) + (rhoR*unR*HR))
        return [Frho, Frhou, FrhoE]

    @_numfluxdict.register(name='hlle')
    def numflux_hlle(self, pdataL, pdataR, dir): # HLLE Riemann solver ; pL[ieq][face]
        rhoL, unL, VL, pL, HL, cL2 = self._derived_fromprim(pdataL, dir)
        rhoR, unR, VR, pR, HR, cR2 = self._derived_fromprim(pdataR, dir)    
        # The HLLE Riemann solver
        etL   = HL-pL/rhoL
        etR   = HR-pR/rhoR
        # Roe's averaging
        Rrho, uRoe, cRoe = self._Roe_average(rhoL, unL, VL, HL, rhoR, unR, VR, HR)
        # max HLL 2 waves "velocities"
        sL = np.minimum(0., np.minimum(uRoe-cRoe, unL-np.sqrt(cL2)))
        sR = np.maximum(0., np.maximum(uRoe+cRoe, unR+np.sqrt(cR2)))
        # final flux
        Frho  = (sR*rhoL*unL - sL*rhoR*unR + sL*sR*(rhoR-rhoL))/(sR-sL)
        Frhou = (sR*((rhoL*unL)*VL + pL*dir) - sL*((rhoR*unR)*VR + pR*dir) + sL*sR*(rhoR*VR-rhoL*VL))/(sR-sL)
        FrhoE = (sR*(rhoL*unL*HL) - sL*(rhoR*unR*HR) + sL*sR*(rhoR*etR-rhoL*etL))/(sR-sL)
        return [Frho, Frhou, FrhoE]

    @_bcdict.register()
    def bc_sym(self, dir, data, param):
        "symmetry boundary condition, for inviscid equations, it is equivalent to a wall, do not need user parameters"
        VL=data[1]
        Vn=_vec_dot_vec(VL,dir)
        VR=VL-2.0*(Vn*dir)
        return [ data[0], VR, data[2] ]

    @_bcdict.register()
    def bc_insub(self, dir, data, param):
        #needed parameters : ptot, rttot
        g   = self.gamma
        gmu = g-1.
        p  = data[2]
        m2 = np.maximum(0., ((param['ptot']/p)**(gmu/g)-1.)*2./gmu)
        rh = param['ptot']/param['rttot']/(1.+.5*gmu*m2)**(1./gmu)
        return [ rh, _sca_mult_vec(-np.sqrt(g*p*m2/rh),dir), p ] 

    @_bcdict.register()
    def bc_insup(self, dir, data, param):
        # needed parameters : ptot, rttot
        g   = self.gamma
        gmu = g-1.
        p  = param['p']
        if 'angle' in param:
            ang = np.deg2rad(param['angle'])
            dir_in = np.full_like(dir, [[np.cos(ang)],[np.sin(ang)]])
        else:
            dir_in = -dir
        m2 = np.maximum(0., ((param['ptot']/p)**(gmu/g)-1.)*2./gmu)
        rh = param['ptot']/param['rttot']/(1.+.5*gmu*m2)**(1./gmu)
        return [rh, _sca_mult_vec(np.sqrt(g*p*m2/rh),dir_in), p]

    @_bcdict.register()
    def bc_outsub(self, dir, data, param):
        return [ data[0], data[1], param['p'] ] 

    @_bcdict.register()
    def bc_outsup(self, dir, data, param):
        return data


# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()

