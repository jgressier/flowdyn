import pytest
#
import numpy as np
#
import flowdyn.mesh2d as mesh2d
import flowdyn.field as field
import flowdyn.integration as integ
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc      as modeldisc
import flowdyn.xnum  as xn
#import flowdyn.solution.euler_riemann as sol



class Test_densitypulse():

    def case_solver(self, nx, ny, xnum, flux):
        meshsim  = mesh2d.unimesh(nx, ny)
        bcper = { 'type': 'per' }
        bclist={ tag:bcper for tag in meshsim.list_of_bctags()} 
        model = euler.euler2d()

        rhs    = modeldisc.fvm2d(model, meshsim, 
                                 num=xnum, numflux=flux, 
                                 bclist=bclist)
        solver = integ.rk3ssp(meshsim, rhs)

        def fuv(x,y):
            return euler.datavector(0.*x+.4, 0.*x+.2)
        def fp(x,y): # gamma = 1.4
            return 0.*x+1.
        def frho(x,y):
            return 1.4 * (1+.2*np.exp(-((x-.5)**2+(y-.5)**2)/(.1)**2))

        xc, yc = meshsim.centers()
        finit = rhs.fdata_fromprim([ frho(xc, yc), fuv(xc, yc), fp(xc, yc) ]) # rho, (u,v), p
        return solver, finit

    def test_variables(self):
        solver, finit = self.case_solver(50, 50, xn.extrapol2d1(), 'centered')
        assert not finit.isnan()
        data = 2*finit.phydata('kinetic-energy') - finit.phydata('density')*(
                 finit.phydata('velocity_x')**2 + finit.phydata('velocity_y')**2)
        error = np.sum(np.abs(data))
        assert error < 1.e-12

    def test_centered(self):
        solver, finit = self.case_solver(50, 50, xn.extrapol2d1(), 'centered')
        endtime = 5.
        cfl     = 2.5
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        rhoavg, rhovar = fsol[-1].stats("density")
        assert rhoavg == pytest.approx(1.408796) # mass conservation
        assert rhovar == pytest.approx(.00115, rel=.01)

    def test_O1_hlle(self):
        solver, finit = self.case_solver(50, 50, xn.extrapol2d1(), 'hlle')
        endtime = 5.
        cfl     = 1.6
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        rhoavg, rhovar = fsol[-1].stats("density")
        assert rhoavg == pytest.approx(1.408796) # mass conservation
        assert rhovar == pytest.approx(5.12e-6, rel=.01)

    def test_O3_hlle(self):
        solver, finit = self.case_solver(50, 50, xn.extrapol2dk(1./3.), 'hlle')
        endtime = 5.
        cfl     = 1.2
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        rhoavg, rhovar = fsol[-1].stats("density")
        assert rhoavg == pytest.approx(1.408796) # mass conservation
        assert rhovar == pytest.approx(5.68e-4, rel=.01)

class TestStraightDuct2d():

    def case_solver(self, nx, ny, xnum, flux, bcL, bcR):
        meshsim  = mesh2d.unimesh(nx, ny)
        bcsym = { 'type': 'sym' }
        bclist={ 'top': bcsym, 'bottom': bcsym,
                 'left': bcL, 'right': bcR } 
        model = euler.euler2d()
        rhs    = modeldisc.fvm2d(model, meshsim, 
                                 num=xnum, numflux=flux, 
                                 bclist=bclist)
        solver = integ.rk3ssp(meshsim, rhs)
        finit = rhs.fdata_fromprim([ 1., [0.8, 0.], 1. ]) # rho, (u,v), p
        return solver, finit

    def test_flow_sub(self):
        endtime = 100.
        cfl     = 1.5
        bcL = { 'type': 'insub',  'ptot': 1.4, 'rttot': 1. }
        bcR = { 'type': 'outsub', 'p': 1. }
        solver, finit = self.case_solver(20, 5, xn.extrapol2d1(), 'hlle', bcL, bcR)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        mach_th = np.sqrt(((bcL['ptot']/bcR['p'])**(1./3.5)-1.)/.2)
        error = np.sqrt(np.sum((fsol[-1].phydata('mach')-mach_th)**2)/fsol[-1].nelem)/mach_th 
        print(fsol[-1].phydata('mach'), mach_th)
        assert error < 1.e-8

    def test_flow_sub(self):
        endtime = 100.
        cfl     = 1.5
        bcL = { 'type': 'insub',  'ptot': 1.4, 'rttot': 1. }
        bcR = { 'type': 'outsub', 'p': 1. }
        solver, finit = self.case_solver(20, 5, xn.extrapol2d1(), 'hlle', bcL, bcR)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        mach_th = np.sqrt(((bcL['ptot']/bcR['p'])**(1./3.5)-1.)/.2)
        error = np.sqrt(np.sum((fsol[-1].phydata('mach')-mach_th)**2)/fsol[-1].nelem)/mach_th 
        assert error < 1.e-8

    def test_flow_sup(self):
        endtime = 100.
        cfl     = 1.5
        bcL = { 'type': 'insup',  'ptot': 2.8, 'rttot': 1., 'p': 1. }
        bcR = { 'type': 'outsup'}
        solver, finit = self.case_solver(20, 5, xn.extrapol2d1(), 'hlle', bcL, bcR)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        mach_th = np.sqrt(((bcL['ptot']/bcL['p'])**(1./3.5)-1.)/.2)
        error = np.sqrt(np.sum((fsol[-1].phydata('mach')-mach_th)**2)/fsol[-1].nelem)/mach_th
        assert error < 1.e-8
