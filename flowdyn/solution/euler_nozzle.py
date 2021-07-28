# -*- coding: utf-8 -*-
"""
test integration methods
"""

import numpy as np
import flowdyn.field           as field
import aerokit.aero.Isentropic as Is
import aerokit.aero.ShockWave  as sw
import aerokit.aero.MassFlow   as mf
import aerokit.aero.nozzle as nz
import aerokit.common.defaultgas as defg

class nozzle():
    """ Define a nozzle

    :param model: define a `modelphy.euler` model, needs the gamma coefficient
    :param section: array of section law
    :param NPR: NPR value (>1); if None (default), can be set with nozzle.set_NPR(NPR)
    :param ref_rttot: additional definition of r*Ttot to complete state (default 1.)
    :param scale_ps: arbitrary scaling of static (and associated total) pressure (default 1. at the outlet)
    """

    def __init__(self, model, section, NPR=None, ref_rttot=1., scale_ps=1.):

        self.model   = model
        self._gam    = self.model.gamma
        self.section = section
        self.ithroat = section.argmin()
        self.AsoAc   = section[-1]/section[self.ithroat]
        self.nozzle  = nz.nozzle(x=None, section=section, gamma=self._gam) #for future version: , ref_rttot=ref_rttot, scale_ps=1.)
        self.NPR0, self.NPRsw, self.NPR1 = self.nozzle.NPR0, self.nozzle.NPRsw, self.nozzle.NPR1
        self._ref_rttot = ref_rttot
        self._scale_ps  = scale_ps
        if NPR:
            self.set_NPR(NPR)
        return

    def set_NPR(self, NPR):
        """ Define Nozzle Pressure Ratio (inlet Ptot over outlet Ps) for this case

        :param NPR: NPR value (>1)

        """
        self.NPR = NPR
        defg.set_gamma(self._gam)
        if NPR < self.NPR0: # flow is fully subsonic
            _Ms = Is.Mach_PiPs(NPR)
            _M  = mf.MachSub_Sigma(self.section*mf.Sigma_Mach(_Ms)/self.section[-1])
            _Pt = 0.*_M + NPR
            _Ps = _Pt/Is.PiPs_Mach(_M)
        else:
            # compute Mach, assumed to be subsonic before throat, supersonic after
            _Minit = 0.*self.section +.5
            _Minit[self.ithroat:] = 2.
            _M  = mf.Mach_Sigma(self.section/self.section[self.ithroat], Mach=_Minit)
            _Pt = NPR+0.*_M
            # CHECK, there is a shock
            # analytical solution for Ms, losses and upstream Mach number of shock wave
            Ms     = nz.Ms_from_AsAc_NPR(self.AsoAc, NPR)
            Ptloss = Is.PiPs_Mach(Ms)/NPR
            Msh    = sw.Mn_Pi_ratio(Ptloss)
            #
            if NPR < self.NPRsw: # throat is choked, there may be a shock
                # redefine curves starting from 'ish' index (closest value of Msh in supersonic flow)
                ish       = np.abs(_M-Msh).argmin()
                _M[ish:]  = mf.MachSub_Sigma(self.section[ish:]*mf.Sigma_Mach(Ms)/self.section[-1])
                _Pt[ish:] = Ptloss*NPR
            _Ps = _Pt/Is.PiPs_Mach(_M)
        #
        self._M  = _M
        self._Pt = _Pt
        self._Ps = _Ps
        return

    def Mach(self):
        """ """
        return self._M

    def Ptot(self):
        """ """
        return self._Ptot

    def Ps(self):
        """ """
        return self._Ps

    def primdata(self):
        """ computes list of rho, u, p data

        """
        p   = self._scale_ps * self._Ps
        rt  = self._ref_rttot/Is.TiTs_Mach(self._M, gamma=self._gam)
        rho = p/rt
        u   = self._M*np.sqrt(self._gam*rt) 
        return [rho, u, p]
        
    def consdata(self):
        """
        """
        return self.model.prim2cons(self.primdata())

    def fdata(self, mesh):
        """
        """
        qcons = self.consdata()
        return field.fdata(self.model, mesh, qcons)

        
