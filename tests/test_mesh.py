import numpy as np
import flowdyn.mesh  as mesh
import flowdyn.modelphy.convection as conv
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
from flowdyn.xnum  import *
from flowdyn.integration import *
import pytest

mymodel = conv.model(1.)

# periodic wave
def init_sinperk(mesh, k):
    return np.sin(2*k*np.pi/mesh.length*mesh.centers())

def test_mesh_uni():
    lmesh = mesh.unimesh(ncell=50, length=1.)
    endtime = 5
    cfl     = 1.
    # extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
    xnum = extrapol3()
    # explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
    tnum  = rk3ssp
    finit = field.fdata(mymodel, lmesh, [ init_sinperk(lmesh, k=2) ] )
    rhs = modeldisc.fvm(mymodel, lmesh, xnum)
    solver = tnum(lmesh, rhs)
    fsol = solver.solve(finit, cfl, [endtime])
    assert not fsol[-1].isnan()

@pytest.mark.parametrize("lratio", [.5, 1., 2.])
def test_mesh_refined(lratio):
    lmesh = mesh.refinedmesh(ncell=50, length=1., ratio=2., nratioa=lratio)
    endtime = 5
    cfl     = 1.
    # extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
    xnum = extrapol3()
    # explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
    tnum  = rk3ssp
    finit = field.fdata(mymodel, lmesh, [ init_sinperk(lmesh, k=2) ] )
    rhs = modeldisc.fvm(mymodel, lmesh, xnum)
    solver = tnum(lmesh, rhs)
    fsol = solver.solve(finit, cfl, [endtime])
    assert not fsol[-1].isnan()

def test_mesh_morphed():
    lmesh = mesh.morphedmesh(ncell=50, length=10., morph=lambda x: x+.3*np.sin(x))
    endtime = 5
    cfl     = 1.
    # extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
    xnum = extrapol3()
    # explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
    tnum  = rk3ssp
    finit = field.fdata(mymodel, lmesh, [ init_sinperk(lmesh, k=2) ] )
    rhs = modeldisc.fvm(mymodel, lmesh, xnum)
    solver = tnum(lmesh, rhs)
    fsol = solver.solve(finit, cfl, [endtime])
    assert not fsol[-1].isnan()