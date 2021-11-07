import numpy as np
import pytest
import flowdyn.mesh  as mesh
import flowdyn.modelphy.convection as conv
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
import flowdyn.xnum  as xnum
import flowdyn.integration as tnum


class monitor_data():

    mesh50 = mesh.unimesh(ncell=50, length=1.)
    convmodel = conv.model(convcoef=1.)
    eulermodel = euler.model()
    xsch = xnum.extrapol3()

    def init_sinperk(self, mesh, k):
        return np.sin(2*k*np.pi/mesh.length*mesh.centers())

@pytest.mark.parametrize("montype", ['residual'])
class Test_Monitor_api(monitor_data):

    def test_frequency(self, montype):
        endtime = 5.
        cfl     = .5
        finit = field.fdata(self.convmodel, self.mesh50, [ self.init_sinperk(self.mesh50, k=4) ] )
        rhs = modeldisc.fvm(self.convmodel, self.mesh50, self.xsch)
        maxit   = 500
        stop_directive = { 'maxit': maxit }
        solver = tnum.rk4(self.mesh50, rhs)
        mondict = {}
        for f in 1, 2, 10, 50, 100:
            monitors = { montype: {'frequency': f}}
            fsol = solver.solve(finit, cfl, [endtime], 
                        stop=stop_directive, monitors=monitors)
            mondict[f] = monitors[montype]['output']
            assert not fsol[-1].isnan()
            assert mondict[f]._it[-1] == maxit
            assert (len(mondict[f]._it)-1)*f == maxit

class Test_Monitor_Euler(monitor_data):

    cfl = 1.2
    bcL = { 'type': 'insub',  'ptot': 1.4, 'rttot': 1. }
    bcR = { 'type': 'outsub', 'p': 1. }

    def test_residual(self):
        rhs = modeldisc.fvm(self.eulermodel, self.mesh50, self.xsch, 
            bcL=self.bcL, bcR=self.bcR)
        finit = rhs.fdata_fromprim([ 1., 0., 1. ]) # rho, u, p
        solver = tnum.rk4(self.mesh50, rhs)
        stop_directive = { 'maxit': 800 }
        monitors = { 'res_euler': {'type': 'residual' ,'frequency': 5}}
        fsol = solver.solve(finit, self.cfl, 
                        stop=stop_directive, monitors=monitors)
        assert not fsol[-1].isnan()
        assert monitors['res_euler']['output'].lastratio() < 1.e-3

    def test_datavg(self):
        rhs = modeldisc.fvm(self.eulermodel, self.mesh50, self.xsch, 
            bcL=self.bcL, bcR=self.bcR)
        finit = rhs.fdata_fromprim([ 1., 0., 1. ]) # rho, u, p
        solver = tnum.rk4(self.mesh50, rhs)
        stop_directive = { 'maxit': 800 }
        monitors = { 'Mach_avg': {'type': 'data_average' , 'data': 'mach', 'frequency': 5}}
        fsol = solver.solve(finit, self.cfl, 
                        stop=stop_directive, monitors=monitors)
        assert not fsol[-1].isnan()
        mach_th = np.sqrt(((self.bcL['ptot']/self.bcR['p'])**(1./3.5)-1.)/.2)
        error = abs(monitors['Mach_avg']['output']._value[-1]-mach_th)/mach_th 
        assert error < 1.e-2