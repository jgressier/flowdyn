import pytest
#
# import numpy as np
import flowdyn.mesh  as mesh
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
from flowdyn.xnum  import *
import flowdyn.integration as integ

mesh100 = mesh.unimesh(ncell=100, length=1.)
mesh50  = mesh.unimesh(ncell=50, length=1.)

model = euler.model()

# initial functions
def fu(x):
    vmag = .01 ; k = 10.
    return vmag*np.exp(-200*(x-.2)**2)*np.sin(2*np.pi*k*x)
def fp(x): # gamma = 1.4
    return (1. + .2*fu(x))**7.  # satisfies C- invariant to make only C+ wave
def frho(x):
    return 1.4 * fp(x)**(1./1.4)

@pytest.mark.parametrize("flux", ["hlle", "hllc"])
def test_acousticpacket_sym(flux):
    endtime = 2.
    cfl     = 0.6
    xnum    = muscl(minmod) 
    tnum    = integ.rk3ssp
    meshsim = mesh100
    xc      = meshsim.centers()
    bcL = { 'type': 'sym'}
    bcR = { 'type': 'sym'}
    rhs = modeldisc.fvm(model, meshsim, xnum, numflux=flux,  bcL=bcL, bcR=bcR)
    finit   = rhs.fdata(model.prim2cons([ frho(xc), fu(xc), fp(xc) ])) # rho, u, p
    solver = tnum(meshsim, rhs)
    fsol = solver.solve(finit, cfl, [endtime])
    assert not fsol[-1].isnan()

def test_ductflow():
    endtime = 100.
    cfl     = 1.2
    xnum    = muscl(minmod) 
    tnum    = integ.rk4
    meshsim = mesh50
    bcL = { 'type': 'insub',  'ptot': 1.4, 'rttot': 1. }
    bcR = { 'type': 'outsub', 'p': 1. }
    rhs = modeldisc.fvm(model, meshsim, xnum, bcL=bcL, bcR=bcR)
    finit = rhs.fdata_fromprim([ 1., 0., 1. ]) # rho, u, p
    solver = tnum(meshsim, rhs)
    fsol = solver.solve(finit, cfl, [endtime])
    assert not fsol[-1].isnan()
    mach_th = np.sqrt(((bcL['ptot']/bcR['p'])**(1./3.5)-1.)/.2)
    error = np.sqrt(np.sum((fsol[-1].phydata('mach')-mach_th)**2)/meshsim.ncell)/mach_th 
    assert error < 1.e-8
