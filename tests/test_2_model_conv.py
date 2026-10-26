import pytest
#
import numpy as np
import flowdyn.mesh  as mesh
import flowdyn.modelphy.convection as conv
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
from flowdyn.xnum import extrapol1, extrapol2, extrapol3, minmod, muscl, vanalbada
from flowdyn.integration import cranknicolson, explicit, implicit, rk2, rk3ssp

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
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()

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
    fsol = solver.solve(finit, cfl, [endtime])
    assert not fsol[-1].isnan()

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
    fsol = solver.solve(finit, cfl, [endtime])
    assert not fsol[-1].isnan()
    assert fsol[-1].average('q') < 1.e-12

def test_numscheme():
    curmesh = mesh50
    endtime = 1
    cfl     = .5
    tnum    = rk3ssp
    for xnum in [ extrapol1(), extrapol2(), extrapol3(), muscl(minmod), muscl(vanalbada) ]:
        finit = field.fdata(mymodel, curmesh, [ init_sinperk(curmesh, k=5) ] )
        rhs = modeldisc.fvm(mymodel, curmesh, xnum)
        solver = tnum(curmesh, rhs)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        assert fsol[-1].average('q') < 1.e-12


def test_rhs_pipeline_does_not_use_stale_intermediates():
    """Ensure each residual evaluation derives all temporaries from its input field."""
    finit = field.fdata(mymodel, mesh50, [init_sinperk(mesh50, k=3)])
    rhs = modeldisc.fvm(mymodel, mesh50, extrapol1())
    expected = [component.copy() for component in rhs.rhs(finit)]

    rhs.qdata = [np.full(mesh50.ncell, np.nan)]
    rhs.pdata = [np.full(mesh50.ncell, np.nan)]
    rhs.grad = [np.full(mesh50.ncell + 1, np.nan)]
    rhs.pL = rhs.pR = [np.full(mesh50.ncell + 1, np.nan)]
    rhs.flux = [np.full(mesh50.ncell + 1, np.nan)]
    rhs.residual = [np.full(mesh50.ncell, np.nan)]

    actual = rhs.rhs(finit)

    np.testing.assert_allclose(actual, expected)
    assert not np.isnan(rhs.pdata[0]).any()
