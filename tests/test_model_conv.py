import pytest
#
import numpy as np
import pyfvm.mesh  as mesh
import pyfvm.modelphy.convection as conv
import pyfvm.modeldisc as modeldisc
import pyfvm.field as field
from pyfvm.xnum  import *
from pyfvm.integration import *

mesh100 = mesh.unimesh(ncell=100, length=1.)
mesh50  = mesh.unimesh(ncell=50, length=1.)

mymodel = conv.model(1.)

# periodic wave
def init_sinperk(mesh, k):
    return np.sin(2*k*np.pi/mesh.length*mesh.centers())

def test_mesh():
    endtime = 5
    cfl     = .5
    # extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
    xnum = extrapol1()
    # explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
    tnum  = explicit
    for curmesh in [ mesh50, mesh100]:
        finit = field.fdata(mymodel, curmesh, [ init_sinperk(curmesh, k=2) ] )
        rhs = modeldisc.fvm(mymodel, curmesh, xnum)
        solver = tnum(curmesh, rhs)
        solver.solve(finit, cfl, [endtime])
    assert 1

@pytest.mark.parametrize("k", [2, 5, 10, 20 ])
def test_wavelength(k):
    curmesh = mesh50
    endtime = 5
    cfl     = .8
    # extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
    xnum = extrapol1()
    # explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
    tnum  = explicit
    finit = field.fdata(mymodel, curmesh, [ init_sinperk(curmesh, k=k) ] )
    rhs = modeldisc.fvm(mymodel, curmesh, xnum)
    solver = tnum(curmesh, rhs)
    solver.solve(finit, cfl, [endtime])
    assert 1

@pytest.mark.parametrize("tnum", [ explicit,rk2, rk3ssp, implicit, cranknicolson ])
def test_integrators(tnum):
    curmesh = mesh50
    endtime = 1
    cfl     = .8
    # extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
    xnum = extrapol1()
    # explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
    finit = field.fdata(mymodel, curmesh, [ init_sinperk(curmesh, k=2) ] )
    rhs = modeldisc.fvm(mymodel, curmesh, xnum)
    solver = tnum(curmesh, rhs)
    solver.solve(finit, cfl, [endtime])
    assert 1

def test_numscheme():
    curmesh = mesh50
    endtime = 1
    cfl     = .5
    tnum    = rk3ssp
    for xnum in [ extrapol1(), extrapol2(), extrapol3(), muscl(minmod), muscl(vanalbada) ]:
        finit = field.fdata(mymodel, curmesh, [ init_sinperk(curmesh, k=5) ] )
        rhs = modeldisc.fvm(mymodel, curmesh, xnum)
        solver = tnum(curmesh, rhs)
        solver.solve(finit, cfl, [endtime])
    assert 1