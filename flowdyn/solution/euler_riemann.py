# -*- coding: utf-8 -*-
"""
test integration methods
"""

import numpy as np
import flowdyn.field           as field
import flowdyn.modelphy.euler  as euler
#import flowdyn.modeldisc       as modeldisc
import aerokit.aero.unsteady1D as uq
import aerokit.instance.riemann as riem

class riemann():
    """ Defines a Riemann problem

        :param model: model for gas
        :param primL: primitive (rho, u, p) data on the left
        :param primR: primitive (rho, u, p) data on the right
    """

    def __init__(self, model: euler.base, primL, primR):

        self.model = model
        self.rhoL, self.uL, self.pL = primL
        self.rhoR, self.uR, self.pR = primR

        gam  = self.model.gamma

        qL = uq.unsteady_state(self.rhoL, self.uL, self.pL, gam)
        qR = uq.unsteady_state(self.rhoR, self.uR, self.pR, gam)
        # Riemann problem
        self.riempb = riem.riemann_pb(qL, qR)

    def primdata(self, mesh, t=None):
        """ Computes and returns primitive data interpolated on mesh at a given time t

        :param mesh: mesh
        :param t: time 

        """
        if t is None:
            xot = np.where(mesh.centers()<0., -1e6, 1e6)
        else:
            xot = mesh.centers()/t
        q = self.riempb.qsol(xot) # 1D data object
        return [q.rho, q.u, q.p]
        
    def consdata(self, mesh, t=None):
        """ Computes and returns primitive data interpolated on mesh at a given time t

        :param mesh: mesh 
        :param t: time

        """
        q = self.primdata(mesh, t)
        return self.model.prim2cons(q)

    def fdata(self, mesh, t=None):
        """

        :param mesh: 
        :param t:  (Default value = None)

        """
        qcons = self.consdata(mesh, t)
        return field.fdata(self.model, mesh, qcons)

    def bcL(self):
        """ """
        return [self.rhoL, self.uL, self.pL]
        
    def bcR(self):
        """ """
        return [self.rhoR, self.uR, self.pR]
        
class Sod_subsonic(riemann):
    """ """
    def __init__(self, model):
        riemann.__init__(self, model, 
                        [1., 0., 1.], [0.125, 0., 0.1])

class Sod_supersonic(riemann):
    """ """
    def __init__(self, model):
        riemann.__init__(self, model,
                        [1., 0., 1.], [0.01, 0., 0.01])

