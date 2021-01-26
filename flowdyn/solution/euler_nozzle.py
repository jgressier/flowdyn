# -*- coding: utf-8 -*-
"""
test integration methods
"""

import numpy as np
import flowdyn.field           as field
import flowdyn.modelphy.euler  as euler
import flowdyn.modeldisc       as modeldisc
import hades.aero.Isentropic as Is
import hades.aero.ShockWave  as sw
import hades.aero.MassFlow   as mf
import hades.aero.nozzle     as nz

class nozzle():

    def __init__(self, model, section):

        self.model   = model
        self.section = section
        self.AsoAc   = section[-1]/np.min(section)
        self.NPR0, self.NPRsw, self.NPR1, self.Msub, self.Msh, self.Msup = nz._NPR_Ms_list(self.AsoAc)

    def set_NPR(NPR):
        if NPR < NPR0:
            _Ms = Is.Mach_PiPs(NPR, gamma=self.model.gamma)
            self._M  = mf.MachSub_Sigma(self.AsoAc*mf.Sigma_Mach(Ma_col)/self.AsoAc[-1], gamma=self.model.gamma)
            self._Pt = 0.*coord_x + 1.
            self._Ps = _Pt/Is.PiPs_Mach(self._M, gamma=self.model.gamma)
        elif NPR < NPRsw:
            _M  = mf.Mach_Sigma(Noz_AoAc, Mach=_Minit)
            #
            # analytical solution for Ms, losses and upstream Mach number of shock wave
            Ms     = nz.Ms_from_AsAc_NPR(target_AoAc, NPR)
            Ptloss = Is.PiPs_Mach(Ms)/NPR
            Msh    = sw.Mn_Pi_ratio(Ptloss)
            #
            # redefine curves starting from 'ish' index (closest value of Msh in supersonic flow)
            ish    = np.abs(_M-Msh).argmin()
            _M[ish:] = mf.MachSub_Sigma(Noz_AoAc[ish:]*mf.Sigma_Mach(Ms)/target_AoAc)
            _Pt[ish:] = Ptloss
            _Ps = _Pt/Is.PiPs_Mach(_M)
        #

    def Mach(self):
        return

    def primdata(self, ):
       return [q.rho, q.u, q.p]
        
    def consdata(self, ):
        return self.model.prim2cons([q.rho, q.u, q.p])

    def fdata(self, mesh, t=None):
        qcons = self.model.prim2cons([q.rho, q.u, q.p])
        return field.fdata(self.model, mesh, qcons)

    def bcL(self):
        return [self.rhoL, self.uL, self.pL]
        
    def bcR(self):
        return [self.rhoR, self.uR, self.pR]
        
