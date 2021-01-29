import pytest
#
# import numpy as np
import flowdyn.mesh  as mesh
import flowdyn.modelphy.euler as euler
import flowdyn.solution.euler_riemann as solR
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
from flowdyn.xnum  import *
import flowdyn.integration as integ

meshsim = mesh.unimesh(ncell=200, length=10., x0=-4.)
meshref = mesh.unimesh(ncell=1000, length=10., x0=-4.)

@pytest.mark.parametrize("case, endtime", [(solR.Sod_subsonic, 2.8), (solR.Sod_supersonic, 2.)])
def test_shocktube(case, endtime):
    model = euler.model()
    sod   = case(model) # sol.Sod_supersonic(model) # 
    bcL  = { 'type': 'dirichlet',  'prim':  sod.bcL() }
    bcR  = { 'type': 'dirichlet',  'prim':  sod.bcR() }
    xnum = muscl(minmod) # 
    rhs = modeldisc.fvm(model, meshsim, xnum, numflux='hllc', bcL=bcL, bcR=bcR)
    solver = integ.rk3ssp(meshsim, rhs)
    # computation
    #
    cfl     = 1.
    finit = sod.fdata(meshsim)
    fsol  = solver.solve(finit, cfl, [endtime])
    fref  = sod.fdata(meshsim, endtime)
    #
    for name in ['density', 'pressure', 'mach']:
        error = np.sqrt(np.sum((fsol[-1].phydata(name)-fref.phydata(name))**2))/np.sum(np.abs(fref.phydata(name)))
        assert error < 1.e-2

@pytest.mark.no_cover
@pytest.mark.parametrize("flux", ["hlle", "hllc"])
def test_shocktube_sodsub(flux):
    model = euler.model()
    sod   = solR.Sod_subsonic(model) # sol.Sod_supersonic(model) # 
    bcL  = { 'type': 'dirichlet',  'prim':  sod.bcL() }
    bcR  = { 'type': 'dirichlet',  'prim':  sod.bcR() }
    xnum = muscl(minmod) # 
    rhs = modeldisc.fvm(model, meshref, xnum, numflux=flux, bcL=bcL, bcR=bcR)
    solver = integ.rk3ssp(meshref, rhs)
    # computation
    #
    endtime = 2.8
    cfl     = 1.
    finit = sod.fdata(meshref)
    fsol  = solver.solve(finit, cfl, [endtime])
    fref  = sod.fdata(meshref, endtime)
    #
    for name in ['density', 'pressure', 'mach']:
        error = np.sqrt(np.sum((fsol[-1].phydata(name)-fref.phydata(name))**2))/np.sum(np.abs(fref.phydata(name)))
        assert error < 5.e-3

@pytest.mark.no_cover
@pytest.mark.parametrize("flux", ["hlle", "hllc"])
def test_shocktube_sodsup(flux):
    model = euler.model()
    sod   = solR.Sod_supersonic(model) # sol.Sod_supersonic(model) # 
    bcL  = { 'type': 'dirichlet',  'prim':  sod.bcL() }
    bcR  = { 'type': 'dirichlet',  'prim':  sod.bcR() }
    xnum = muscl(minmod) # 
    rhs = modeldisc.fvm(model, meshref, xnum, numflux=flux, bcL=bcL, bcR=bcR)
    solver = integ.rk3ssp(meshref, rhs)
    # computation
    #
    endtime = 2.
    cfl     = 1.
    finit = sod.fdata(meshref)
    fsol  = solver.solve(finit, cfl, [endtime])
    fref  = sod.fdata(meshref, endtime)
    #
    for name in ['density', 'pressure', 'mach']:
        error = np.sqrt(np.sum((fsol[-1].phydata(name)-fref.phydata(name))**2))/np.sum(np.abs(fref.phydata(name)))
        assert error < 5.e-3
