import numpy as np
import pytest
import flowdyn.mesh  as mesh
import flowdyn.modelphy.convection as conv
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
import flowdyn.xnum  as xnum
import flowdyn.integration as tnum

# @pytest.fixture
# def convmodel():
#     return conv.model(1.)



class integration_data():

    curmesh = mesh.unimesh(ncell=50, length=1.)
    convmodel = conv.model(convcoef=1.)

    def init_sinperk(self, mesh, k):
        return np.sin(2*k*np.pi/mesh.length*mesh.centers())

class Test_solve(integration_data):

    xsch = xnum.extrapol3()

    def test_checkend_tottime(self):
        endtime = 5.
        cfl     = .5
        stop_directive = { 'tottime': endtime }
        finit = field.fdata(self.convmodel, self.curmesh, [ self.init_sinperk(self.curmesh, k=4) ] )
        rhs = modeldisc.fvm(self.convmodel, self.curmesh, self.xsch)
        solver = tnum.rk4(self.curmesh, rhs)
        nsol = 11
        tsave = np.linspace(0, 2*endtime, nsol, endpoint=True)
        fsol = solver.solve(finit, cfl, tsave, stop=stop_directive)
        assert len(fsol) < nsol # end before expected by tsave
        assert not fsol[-1].isnan()
        assert fsol[-1].time < 2*endtime

    def test_checkend_maxit(self):
        endtime = 5.
        cfl     = .5
        maxit   = 100
        stop_directive = { 'maxit': maxit }
        finit = field.fdata(self.convmodel, self.curmesh, [ self.init_sinperk(self.curmesh, k=4) ] )
        rhs = modeldisc.fvm(self.convmodel, self.curmesh, self.xsch)
        solver = tnum.rk4(self.curmesh, rhs)
        nsol = 11
        tsave = np.linspace(0, endtime, nsol, endpoint=True)
        fsol = solver.solve(finit, cfl, tsave, stop=stop_directive)
        assert len(fsol) < nsol # end before expected by tsave
        assert not fsol[-1].isnan()
        assert fsol[-1].time < endtime
        assert solver.nit() == 100

    def test_interpolation_regression(self):
        endtime = 5
        cfl     = .5
        finit = field.fdata(self.convmodel, self.curmesh, [ self.init_sinperk(self.curmesh, k=4) ] )
        rhs = modeldisc.fvm(self.convmodel, self.curmesh, self.xsch)
        solver = tnum.rk4(self.curmesh, rhs)
        nsol = 5
        tsave = np.linspace(0, endtime, nsol, endpoint=True)
        fsol = solver.solve(finit, cfl, tsave)
        fleg = solver.solve_legacy(finit, cfl, tsave)
        for fs, fl in zip(fsol, fleg):
            assert not fs.isnan()
            diff = fs.diff(fl)
            assert diff.time == pytest.approx(0., abs=1.e-12)
            for d in diff.data:
                assert np.average(np.abs(d)) == pytest.approx(0., abs=1.e-6)


@pytest.mark.parametrize("tmeth", tnum.List_Explicit_Integrators + tnum.List_Implicit_Integrators)
class Test_integrators(integration_data):

    def test_conv_it_perf(self, tmeth):
        endtime = 5
        cfl     = .8
        xsch = xnum.extrapol3()
        finit = field.fdata(self.convmodel, self.curmesh, [ self.init_sinperk(self.curmesh, k=4) ] )
        rhs = modeldisc.fvm(self.convmodel, self.curmesh, xsch)
        solver = tmeth(self.curmesh, rhs)
        fsol = solver.solve(finit, cfl, [endtime])
        assert solver.nit() > 300
        assert solver.nit() < 400
        maxperf = 40. if tmeth in tnum.List_Explicit_Integrators else 100.
        assert solver.perf_micros() < maxperf
        assert not fsol[-1].isnan()

@pytest.mark.parametrize("tmeth", tnum.List_LSRK_Integrators)
class Test_integrators_explicit(integration_data):

    def test_cflmax(self, tmeth):
        solver = tmeth(None, None)
        assert solver.cflmax() > 3.

