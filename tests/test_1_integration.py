import numpy as np
import pytest
import flowdyn.mesh  as mesh
import flowdyn.modelphy.convection as conv
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
from flowdyn.xnum  import *
from flowdyn.integration import *

mesh100 = mesh.unimesh(ncell=100, length=1.)
mesh50  = mesh.unimesh(ncell=50, length=1.)

mymodel = conv.model(1.)

# periodic wave
def init_sinperk(mesh, k):
    return np.sin(2*k*np.pi/mesh.length*mesh.centers())


@pytest.mark.parametrize("tnum", List_Explicit_Integrators + List_Implicit_Integrators)
def test_integrators_conv(tnum):
    curmesh = mesh50
    endtime = 5
    cfl     = .8
    # extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
    xnum = extrapol3()
    finit = field.fdata(mymodel, curmesh, [ init_sinperk(curmesh, k=4) ] )
    rhs = modeldisc.fvm(mymodel, curmesh, xnum)
    solver = tnum(curmesh, rhs)
    solver.solve(finit, cfl, [endtime])
    assert 1

