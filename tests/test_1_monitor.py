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

    curmesh = mesh.unimesh(ncell=50, length=1.)
    convmodel = conv.model(convcoef=1.)
    eulermodel = euler.model()

    def init_sinperk(self, mesh, k):
        return np.sin(2*k*np.pi/mesh.length*mesh.centers())

@pytest.mark.parametrize("montype", ['residual'])
class Test_Monitor_api(monitor_data):

    xsch = xnum.extrapol3()

    def test_frequency(self, montype):
        endtime = 5.
        cfl     = .5
        finit = field.fdata(self.convmodel, self.curmesh, [ self.init_sinperk(self.curmesh, k=4) ] )
        rhs = modeldisc.fvm(self.convmodel, self.curmesh, self.xsch)
        maxit   = 500
        stop_directive = { 'maxit': maxit }
        solver = tnum.rk4(self.curmesh, rhs)
        mondict = {}
        for f in 1, 2, 10, 50, 100:
            monitors = { montype: {'frequency': f}}
            fsol = solver.solve(finit, cfl, [endtime], 
                        stop=stop_directive, monitors=monitors)
            mondict[f] = monitors[montype]['output']
            assert not fsol[-1].isnan()
            assert mondict[f]._it[-1] == maxit
            assert (len(mondict[f]._it)-1)*f == maxit

