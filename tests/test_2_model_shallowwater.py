import numpy as np
import pytest
#
# import numpy as np
import flowdyn.mesh  as mesh
import flowdyn.modelphy.shallowwater as shw
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
from flowdyn.xnum  import *
import flowdyn.integration as integ

# initial functions
def waterdrop(x, hmax, umean=0.): # return h and h*u
    h = .1+hmax*(-np.exp(-((x-.2)/0.1)**2))
    return [ h, h*umean]

@pytest.mark.parametrize("flux", ["rusanov", "centered", "hll"])
def test_sym_flux(flux):
    endtime = 1.
    cfl     = 0.6
    xnum    = muscl(minmod) 
    tnum    = integ.rk3ssp
    meshsim = mesh.unimesh(ncell=100, length=1.)
    xc      = meshsim.centers()
    bcL = { 'type': 'sym'}
    bcR = { 'type': 'sym'}
    model  = shw.shallowwater1d()
    model10 = shw.shallowwater1d(g=10.)
    rhs = modeldisc.fvm(model, meshsim, xnum, numflux=flux,  bcL=bcL, bcR=bcR)
    finit   = rhs.fdata(model.prim2cons(waterdrop(xc, .01))) # rho, u, p
    solver = tnum(meshsim, rhs)
    fsol = solver.solve(finit, cfl, [endtime])
    assert not fsol[-1].isnan()
    avg, var = fsol[-1].stats('height')
    assert avg == pytest.approx(0.098232, rel=1.e-4)
    assert var >= 6.5e-6
