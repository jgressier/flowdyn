import pytest
#
# import numpy as np
import flowdyn.mesh  as mesh
import flowdyn.modelphy.euler as euler
import flowdyn.solution.euler_riemann as solR
import flowdyn.solution.euler_nozzle  as solN
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
def test_shocktube_sodsub_th(flux):
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
def test_shocktube_sodsup_th(flux):
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

@pytest.mark.parametrize("NPR", [1.05, 1.1])
def test_nozzle(NPR):
    def S(x): # section law, throat is at x=5
        return 1.-.5*np.exp(-.5*(x-5.)**2)

    model    = euler.nozzle(sectionlaw=S)
    meshsim  = mesh.unimesh(ncell=50,  length=10.)
    nozz     = solN.nozzle(model, S(meshsim.centers()), NPR=NPR)
    fref     = nozz.fdata(meshsim)
    # BC
    bcL = { 'type': 'insub',  'ptot': NPR, 'rttot': 1. }
    bcR = { 'type': 'outsub', 'p': 1. }
    #
    rhs = modeldisc.fvm(model, meshsim, muscl(vanleer), bcL=bcL, bcR=bcR)
    solver = integ.rk3ssp(meshsim, rhs)
    # computation
    endtime = 100.
    cfl     = .8
    finit   = rhs.fdata_fromprim([  1., 0.1, 1. ]) # rho, u, p
    fsol    = solver.solve(finit, cfl, [endtime])
    # error
    error = np.sqrt(np.sum((fsol[-1].phydata('mach')-fref.phydata('mach'))**2))/np.sum(np.abs(fref.phydata('mach')))
    assert error < 5.e-2

@pytest.mark.no_cover
@pytest.mark.parametrize("gam, NPR", [(1.4, 1.05), (1.4, 1.1), (1.35, 1.1)])
def test_nozzle_th(gam, NPR):
    def S(x): # section law, throat is at x=5
        return 1.-.5*np.exp(-.5*(x-5.)**2)

    model    = euler.nozzle(sectionlaw=S, gamma=gam)
    meshsim  = mesh.unimesh(ncell=200,  length=10.)
    nozz     = solN.nozzle(model, S(meshsim.centers()), NPR=NPR)
    fref     = nozz.fdata(meshsim)
    # BC
    bcL = { 'type': 'insub',  'ptot': NPR, 'rttot': 1. }
    bcR = { 'type': 'outsub', 'p': 1. }
    #
    rhs = modeldisc.fvm(model, meshsim, muscl(vanalbada), bcL=bcL, bcR=bcR)
    solver = integ.rk3ssp(meshsim, rhs)
    # computation
    endtime = 100.
    cfl     = .8
    finit   = rhs.fdata_fromprim([  1., 0.1, 1. ]) # rho, u, p
    fsol    = solver.solve(finit, cfl, [endtime])
    # error
    error = np.sqrt(np.sum((fsol[-1].phydata('mach')-fref.phydata('mach'))**2))/np.sum(np.abs(fref.phydata('mach')))
    assert error < 1.e-2