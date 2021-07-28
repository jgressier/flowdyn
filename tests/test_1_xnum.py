import numpy as np
import pytest
import flowdyn.mesh  as mesh
import flowdyn.modelphy.convection as conv
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
from flowdyn.xnum  import *
from flowdyn.integration import *


@pytest.mark.parametrize("limiter", [minmod, vanalbada, vanleer, superbee])
class Test_limiter_conv():

    mesh100 = mesh.unimesh(ncell=100, length=1.)
    mesh50  = mesh.unimesh(ncell=50, length=1.)

    mymodel = conv.model(1.)

    # periodic wave
    def init_sinperk(self, mesh, k):
        return np.sin(2*k*np.pi/mesh.length*mesh.centers())

    def test_0_tvd(self, limiter):
        slope = limiter(-2., 1.)  # tvd: 0 if opposite signs
        assert slope == 0.
        slope = limiter(3., 3.)   # consistency
        assert slope == 3.
        slope = limiter(1., 10.)  # tvd: max is double of smallest
        assert slope <= 2.

    def test_muscl_limiters(self, limiter):
        curmesh = self.mesh50
        endtime = 5
        cfl     = .8
        # extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
        xnum = muscl(limiter)
        finit = field.fdata(self.mymodel, curmesh, [ self.init_sinperk(curmesh, k=4) ] )
        rhs = modeldisc.fvm(self.mymodel, curmesh, xnum)
        solver = rk3ssp(curmesh, rhs)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()
        avg, var = fsol[-1].stats('q')
        #varref = { }
        assert avg == pytest.approx(0., abs=1.e-12)
        #assert var == pytest.approx(.0042, rel=1.e-2)

