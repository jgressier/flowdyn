# -*- coding: utf-8 -*-
"""
test integration methods
"""

import numpy as np
import flowdyn.field           as field
import flowdyn.modelphy.euler  as euler
import flowdyn.modeldisc       as modeldisc
import hades.aero.unsteady1D as uq
import hades.aero.riemann    as riem

class riemann():

    def __init__(self, model, primL, primR):

        self.model = model
        self.rhoL, self.uL, self.pL = primL
        self.rhoR, self.uR, self.pR = primR

        gam  = self.model.gamma

        qL = uq.unsteady_state(self.rhoL, self.uL, self.pL, gam)
        qR = uq.unsteady_state(self.rhoR, self.uR, self.pR, gam)
        # Riemann problem
        self.riempb = riem.riemann_pb(qL, qR)

    def primdata(self, mesh, t):
        if not t:
            xot = np.where(mesh.centers()<0., -1e6, 1e6)
        else:
            xot = mesh.centers()/t
        q = self.riempb.qsol(xot)
        return [q.rho, q.u, q.p]
        
    def consdata(self, mesh, t):
        if not t:
            xot = np.where(mesh.centers()<0., -1e6, 1e6)
        else:
            xot = mesh.centers()/t
        q = self.riempb.qsol(xot)
        return self.model.prim2cons([q.rho, q.u, q.p])

    def fdata(self, mesh, t=None):
        if t==None:
            xot = np.where(mesh.centers()<0., -1e6, 1e6)
        else:
            xot = mesh.centers()/t
        q = self.riempb.qsol(xot)
        qcons = self.model.prim2cons([q.rho, q.u, q.p])
        return field.fdata(self.model, mesh, qcons)

    def bcL(self):
        return [self.rhoL, self.uL, self.pL]
        
    def bcR(self):
        return [self.rhoR, self.uR, self.pR]
        
class Sod_subsonic(riemann):
    def __init__(self, model):
        riemann.__init__(self, model, 
                        [1., 0., 1.], [0.125, 0., 0.1])

class Sod_supersonic(riemann):
    def __init__(self, model):
        riemann.__init__(self, model,
                        [1., 0., 1.], [0.01, 0., 0.01])

