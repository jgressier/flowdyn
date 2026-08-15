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
    """Define a one-dimensional Euler Riemann problem.

    Args:
        model: Euler model for the gas.
        primL: Left primitive state ``[rho, u, p]``.
        primR: Right primitive state ``[rho, u, p]``.
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
        """Compute primitive data on a mesh at a given time.

        Args:
            mesh: Mesh on which to evaluate the solution.
            t: Evaluation time. If omitted, return the initial discontinuity.

        Returns:
            Primitive state ``[rho, u, p]`` evaluated at cell centers.
        """
        if t is None:
            xot = np.where(mesh.centers()<0., -1e6, 1e6)
        else:
            xot = mesh.centers()/t
        q = self.riempb.qsol(xot) # 1D data object
        return [q.rho, q.u, q.p]
        
    def consdata(self, mesh, t=None):
        """Compute conservative data on a mesh at a given time.

        Args:
            mesh: Mesh on which to evaluate the solution.
            t: Evaluation time. If omitted, return the initial discontinuity.

        Returns:
            Conservative state evaluated at cell centers.
        """
        q = self.primdata(mesh, t)
        return self.model.prim2cons(q)

    def fdata(self, mesh, t=None):
        """Build a Flowdyn field containing the Riemann solution.

        Args:
            mesh: Mesh on which to evaluate the solution.
            t: Evaluation time. If omitted, return the initial discontinuity.

        Returns:
            Field containing the conservative solution.
        """
        qcons = self.consdata(mesh, t)
        return field.fdata(self.model, mesh, qcons)

    def bcL(self):
        """Return the left primitive state."""
        return [self.rhoL, self.uL, self.pL]
        
    def bcR(self):
        """Return the right primitive state."""
        return [self.rhoR, self.uR, self.pR]
        
class Sod_subsonic(riemann):
    """Define the standard subsonic Sod shock-tube problem."""
    def __init__(self, model):
        riemann.__init__(self, model, 
                        [1., 0., 1.], [0.125, 0., 0.1])

class Sod_supersonic(riemann):
    """Define a supersonic Sod-like shock-tube problem."""
    def __init__(self, model):
        riemann.__init__(self, model,
                        [1., 0., 1.], [0.01, 0., 0.01])
