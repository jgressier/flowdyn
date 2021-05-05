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

@pytest.mark.parametrize("tnum", tnum.List_Explicit_Integrators + tnum.List_Implicit_Integrators)
class Test_integrators():

    # periodic wave
    def init_sinperk(self, mesh, k):
        return np.sin(2*k*np.pi/mesh.length*mesh.centers())

    def test_integrators_conv(self, tnum):
        curmesh = mesh.unimesh(ncell=50, length=1.)
        endtime = 5
        cfl     = .8
        convmodel = conv.model(convcoef=1.)
        # extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
        xsch = xnum.extrapol3()
        finit = field.fdata(convmodel, curmesh, [ self.init_sinperk(curmesh, k=4) ] )
        rhs = modeldisc.fvm(convmodel, curmesh, xsch)
        solver = tnum(curmesh, rhs)
        fsol = solver.solve(finit, cfl, [endtime])
        assert not fsol[-1].isnan()

