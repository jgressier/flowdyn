import pytest
#
import numpy as np
#
import flowdyn.mesh2d as mesh2d
import flowdyn.field as field
import flowdyn.integration as integ
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc      as modeldisc
from flowdyn.xnum  import *
#import flowdyn.solution.euler_riemann as sol

def test_densitypulse():

    nx = 50
    ny = 50
    meshsim  = mesh2d.unimesh(nx, ny)

    bcper = { 'type': 'per' }
    bclist={ tag:bcper for tag in meshsim.list_of_bctags()} 
    model = euler.euler2d()

    rhs    = modeldisc.fvm2d(model, meshsim, num=extrapol2d1(), numflux='centered', bclist=bclist)

    solver = integ.rk3ssp(meshsim, rhs)

    def fuv(x,y):
        return euler.datavector(0.*x+.4, 0.*x+.2)
    def fp(x,y): # gamma = 1.4
        return 0.*x+1.
    def frho(x,y):
        return 1.4 * (1+.2*np.exp(-((x-.5)**2+(y-.5)**2)/(.1)**2))

    endtime = 5.
    cfl     = 2.5
    xc, yc = meshsim.centers()
    finit = rhs.fdata_fromprim([ frho(xc, yc), fuv(xc, yc), fp(xc, yc) ]) # rho, (u,v), p
    fsol = solver.solve(finit, cfl, [endtime])
    assert not fsol[-1].isnan()
