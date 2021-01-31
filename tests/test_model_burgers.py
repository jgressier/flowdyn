import numpy as np
import flowdyn.mesh  as mesh
import flowdyn.modelphy.burgers as burgers
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
from flowdyn.xnum  import *
import flowdyn.integration as integ

mesh100 = mesh.unimesh(ncell=100, length=5.)
mesh50  = mesh.unimesh(ncell=50, length=5.)

mymodel = burgers.model()

def init_step(mesh, ul, ur):
    step = np.zeros(len(mesh.centers()))
    xr   = 1.0
    x    = mesh.centers()
    for i in range(len(x)):
        if x[i] < xr:
            step[i] = ul
        elif xr <= x[i] <= 2.0:
            step[i] = 3.0-x[i] 
        elif x[i] > 2.0:
            step[i] = ur
    return step

def init_sin(mesh):
    k = 2 # nombre d'onde
    omega = k*np.pi/mesh.length
    return .2+np.sin(omega*mesh.centers())

def test_compression_prop():
    endtime = 2.
    cfl     = 0.5
    xnum    = muscl(minmod) 
    tnum    = integ.rk3ssp
    thismesh = mesh100
    thisprim = [init_step(thismesh, 2, .5)]
    thiscons = field.fdata(mymodel, thismesh, thisprim)
    bcL = { 'type': 'dirichlet', 'prim': thisprim[0]  }
    bcR = { 'type': 'dirichlet', 'prim': thisprim[-1] }
    rhs = modeldisc.fvm(mymodel, thismesh, xnum, bcL=bcL, bcR=bcR)
    solver = tnum(thismesh, rhs)
    solver.solve(thiscons, cfl, [endtime])
    assert 1

def test_compression_centered():
    endtime = 2.
    cfl     = .5
    xnum    = muscl(minmod) 
    tnum    = integ.rk3ssp
    thismesh = mesh100
    thisprim = [init_step(thismesh, 2, -1.)]
    thiscons = field.fdata(mymodel, thismesh, thisprim)
    bcL = { 'type': 'dirichlet', 'prim': thisprim[0]  }
    bcR = { 'type': 'dirichlet', 'prim': thisprim[-1] }
    rhs = modeldisc.fvm(mymodel, thismesh, xnum, bcL=bcL, bcR=bcR)
    solver = tnum(thismesh, rhs)
    solver.solve(thiscons, cfl, [endtime])
    assert 1

def test_expansion():
    endtime = 2.
    cfl     = 0.5
    xnum    = muscl(minmod) 
    tnum    = integ.rk3ssp
    thismesh = mesh100
    thisprim = [init_step(thismesh, 1., 2.)]
    thiscons = field.fdata(mymodel, thismesh, thisprim)
    bcL = { 'type': 'dirichlet', 'prim': thisprim[0]  }
    bcR = { 'type': 'dirichlet', 'prim': thisprim[-1] }
    rhs = modeldisc.fvm(mymodel, thismesh, xnum, bcL=bcL, bcR=bcR)
    solver = tnum(thismesh, rhs)
    solver.solve(thiscons, cfl, [endtime])
    assert 1

def test_sin_bcper():
    endtime = 2.
    cfl     = 0.5
    xnum    = muscl(minmod) 
    tnum    = integ.rk3ssp
    thismesh = mesh100
    thisprim = [ init_sin(thismesh) ]
    thiscons = field.fdata(mymodel, thismesh, thisprim)
    bcL = { 'type': 'per' }
    bcR = { 'type': 'per' }
    rhs = modeldisc.fvm(mymodel, thismesh, xnum, bcL=bcL, bcR=bcR)
    solver = tnum(thismesh, rhs)
    solver.solve(thiscons, cfl, [endtime])
    assert 1

