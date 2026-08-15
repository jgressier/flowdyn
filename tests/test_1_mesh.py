import numpy as np
import flowdyn.mesh  as mesh
import flowdyn.mesh2d as mesh2d
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

def test_mesh_uni_repr():
    lmesh = mesh.unimesh(ncell=4, length=2.)

    assert repr(lmesh) == (
        "length : 2.0\n"
        "ncell  : 4\n"
        "min dx : 0.5\n"
        "max dx : 0.5"
    )

def test_discretization_copy_preserves_configuration():
    lmesh = mesh.unimesh(ncell=4)
    bcL = {'type': 'dirichlet', 'prim': [1.]}
    rhs = modeldisc.fvm(mymodel, lmesh, extrapol1(), numflux='upwind', bcL=bcL)

    copied = rhs.copy()

    assert type(copied) is type(rhs)
    assert copied.numflux == rhs.numflux
    assert copied.bcL == rhs.bcL
    copied.bcL['type'] = 'changed'
    assert rhs.bcL['type'] == 'dirichlet'

@pytest.mark.parametrize("kwargs", [
    {"ncell": 0},
    {"ncell": 2.5},
    {"length": 0.},
    {"length": np.inf},
])
def test_mesh_uni_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        mesh.unimesh(**kwargs)

def test_mesh2d_repr():
    lmesh = mesh2d.unimesh(nx=4, ny=2, lx=2., ly=1.)
    assert repr(lmesh) == (
        "mesh object: mesh2d\n"
        "dimensions : 4 x 2\n"
        "lengths : 2.0 x 1.0\n"
        "cell sizes : 0.5 x 0.5"
    )

@pytest.mark.parametrize("kwargs", [
    {"nx": 0, "ny": 2},
    {"nx": 2, "ny": -1},
    {"nx": 2, "ny": 2, "lx": 0.},
    {"nx": 2, "ny": 2, "ly": np.nan},
])
def test_mesh2d_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        mesh2d.unimesh(**kwargs)

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
