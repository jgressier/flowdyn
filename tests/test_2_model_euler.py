import pytest
#
import numpy as np
import flowdyn.mesh  as mesh
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
from flowdyn.xnum  import *
import flowdyn.integration as integ

class euler_w_mesh():
    mesh100 = mesh.unimesh(ncell=100, length=1.)
    mesh50  = mesh.unimesh(ncell=50, length=1.)
    mesh20  = mesh.unimesh(ncell=20, length=1.)

class eulerstd(euler_w_mesh):
    model = euler.euler1d()
    # initial functions
    def fu(self, x):
        vmag = .01 ; k = 10.
        return vmag*np.exp(-200*(x-.2)**2)*np.sin(2*np.pi*k*x)
    def fp(self, x): # gamma = 1.4
        return (1. + .2*self.fu(x))**7.  # satisfies C- invariant to make only C+ wave
    def frho(self, x):
        return 1.4 * self.fp(x)**(1./1.4)

class TestAcoustic(eulerstd):

    @pytest.mark.parametrize("flux", ["hlle", "hllc", "centered", "centeredmassflow"])
    def test_acousticpacket_flux(self, flux):
        endtime = 2.
        cfl     = 0.6
        xnum    = muscl(minmod) 
        tnum    = integ.rk3ssp
        meshsim = self.mesh100
        xc      = meshsim.centers()
        bcL = { 'type': 'sym'}
        bcR = { 'type': 'sym'}
        rhs = modeldisc.fvm(self.model, meshsim, xnum, numflux=flux,  bcL=bcL, bcR=bcR)
        finit   = rhs.fdata(self.model.prim2cons([ self.frho(xc), self.fu(xc), self.fp(xc) ])) # rho, u, p
        solver = tnum(meshsim, rhs)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()

    @pytest.mark.parametrize("bcout", ["sym", "outsub", "outsub_nrcbc"])
    def test_acousticpacket_bcout(self, bcout):
        endtime = 2.
        cfl     = 0.6
        xnum    = muscl(minmod) 
        tnum    = integ.rk3ssp
        meshsim = self.mesh100
        xc      = meshsim.centers()
        bcL = { 'type': 'sym'}
        bcR = { 'type': bcout, 'p': 1.} # 'p' will be ignored if unused
        rhs = modeldisc.fvm(self.model, meshsim, xnum, numflux='hllc',  bcL=bcL, bcR=bcR)
        finit   = rhs.fdata(self.model.prim2cons([ self.frho(xc), self.fu(xc), self.fp(xc) ])) # rho, u, p
        solver = tnum(meshsim, rhs)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()

class TestStraightDuct(eulerstd):

    #avoid outsub_nrcbc because of bad convergence
    @pytest.mark.parametrize("bcout", ["outsub_prim", "outsub_qtot", "outsub_rh"]) 
    @pytest.mark.parametrize("bcin", ["insub", "insub_cbc"]) 
    def test_flow_sub(self, bcin, bcout):
        endtime = 100.
        cfl     = 1.2
        xnum    = muscl(minmod) 
        tnum    = integ.rk4
        meshsim = self.mesh20
        bcL = { 'type': bcin,  'ptot': 1.4, 'rttot': 1. }
        bcR = { 'type': bcout, 'p': 1. }
        rhs = modeldisc.fvm(self.model, meshsim, xnum, bcL=bcL, bcR=bcR)
        finit = rhs.fdata_fromprim([ 1., 0., 1. ]) # rho, u, p
        solver = tnum(meshsim, rhs)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        mach_th = np.sqrt(((bcL['ptot']/bcR['p'])**(1./3.5)-1.)/.2)
        error = np.sqrt(np.sum((fsol[-1].phydata('mach')-mach_th)**2)/meshsim.ncell)/mach_th 
        assert error < 1.e-8

    def test_flow_sup(self):
        endtime = 100.
        cfl     = 1.2
        xnum    = muscl(minmod) 
        tnum    = integ.rk4
        meshsim = self.mesh20
        bcL = { 'type': 'insup',  'ptot': 3., 'rttot': 1., 'p': 1.}
        bcR = { 'type': 'outsup' }
        rhs = modeldisc.fvm(self.model, meshsim, xnum, bcL=bcL, bcR=bcR)
        finit = rhs.fdata_fromprim([ 1., 3., 1. ]) # rho, u, p
        solver = tnum(meshsim, rhs)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        mach_th = np.sqrt(((bcL['ptot']/bcL['p'])**(1./3.5)-1.)/.2)
        error = np.sqrt(np.sum((fsol[-1].phydata('mach')-mach_th)**2)/meshsim.ncell)/mach_th 
        assert error < 1.e-8

    def test_variables(self):
        bcL = { 'type': 'insub',  'ptot': 1.4, 'rttot': 1. }
        bcR = { 'type': 'outsub', 'p': 1. }
        meshsim = self.mesh100
        rhs = modeldisc.fvm(self.model, meshsim, extrapol1(), bcL=bcL, bcR=bcR)
        xc  = meshsim.centers()
        finit = rhs.fdata_fromprim([ self.frho(xc), self.fu(xc), self.fp(xc) ]) # rho, u, p
        data = finit.phydata('asound')*finit.phydata('mach')-finit.phydata('velocity')
        error = np.sum(np.abs(data))
        assert error < 1.e-12
        data = finit.phydata('velocity')*finit.phydata('density')-finit.phydata('massflow')
        error = np.sum(np.abs(data))
        assert error < 1.e-12
        data = finit.phydata('htot')-finit.phydata('enthalpy')-finit.phydata('kinetic-energy')-finit.phydata('massflow')
        error = np.sum(np.abs(data))
        # entropy should be constant (variance = 0)
        __, error = finit.stats('entropy')
        assert error < 1.e-12

    def test_fanno(self):
        endtime = 100.
        cfl     = 1.2
        xnum    = muscl(minmod) 
        tnum    = integ.rk4
        meshsim = self.mesh20
        modelfanno = euler.model(source=[None, lambda x,q:-.2, None])
        bcL = { 'type': 'insub',  'ptot': 1.4, 'rttot': 1. }
        bcR = { 'type': 'outsub', 'p': 1. }
        rhs = modeldisc.fvm(modelfanno, meshsim, xnum, bcL=bcL, bcR=bcR)
        finit = rhs.fdata_fromprim([ 1., 0., 1. ]) # rho, u, p
        solver = tnum(meshsim, rhs)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        avg, var = fsol[-1].stats('ptot')
        assert avg == pytest.approx(1.2904, rel=1.e-4)
        assert var == pytest.approx(.0042, rel=1.e-2)

    def test_rayleigh(self):
        endtime = 100.
        cfl     = 1.2
        xnum    = muscl(minmod) 
        tnum    = integ.rk4
        meshsim = self.mesh20
        modelrayleigh = euler.model(source=[None, None, lambda x,q:2.])
        bcL = { 'type': 'insub',  'ptot': 1.4, 'rttot': 1. }
        bcR = { 'type': 'outsub', 'p': 1. }
        rhs = modeldisc.fvm(modelrayleigh, meshsim, xnum, bcL=bcL, bcR=bcR)
        finit = rhs.fdata_fromprim([ 1., 0., 1. ]) # rho, u, p
        solver = tnum(meshsim, rhs)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        avg, var = fsol[-1].stats('rttot')
        assert avg == pytest.approx(1.6139, rel=1.e-4)
        assert var == pytest.approx(0.1242, rel=1.e-2)