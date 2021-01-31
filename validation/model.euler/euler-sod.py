# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
import cProfile
from pylab import *
import numpy as np 

from flowdyn.mesh  import *
from flowdyn.field import *
from flowdyn.xnum  import *
from flowdyn.integration import *
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc      as modeldisc
import flowdyn.solution.euler_riemann as sol

meshsim  = unimesh(ncell=200,  length=10., x0=-4.)
meshref  = unimesh(ncell=1000, length=10., x0=-4.)

model1 = euler.model()
model2 = euler.model()
sod   = sol.Sod_subsonic(model1) # sol.Sod_supersonic(model1) # 

bcL  = { 'type': 'dirichlet',  'prim':  sod.bcL() }
bcR  = { 'type': 'dirichlet',  'prim':  sod.bcR() }
xnum1 = muscl(minmod) # 
xnum2 = muscl(vanalbada) # 

rhs1 = modeldisc.fvm(model1, meshsim, xnum1, numflux='hllc', bcL=bcL, bcR=bcR)
solver1 = rk3ssp(meshsim, rhs1)
rhs2 = modeldisc.fvm(model2, meshsim, xnum1, numflux='hlle', bcL=bcL, bcR=bcR)
solver2 = rk3ssp(meshsim, rhs2)

# computation
#
endtime = 2.8 # 2.8 for subsonic, 2.8 for supersonic
cfl     = 1.

finit = sod.fdata(meshsim)

fsol1 = solver1.solve(finit, cfl, [endtime])
solver1.show_perf()
fsol2 = solver2.solve(finit, cfl, [endtime])
solver2.show_perf()

# Figure / Plot

fref = sod.fdata(meshref, endtime)

for name in ['density', 'pressure', 'mach']:
    fig = figure(figsize=(10,8))
    ylabel(name)
    grid(linestyle='--', color='0.5')
    #finit.plot(name, 'k-.')
    fref.plot(name, 'k-')
    fsol1[0].plot(name, 'b-')
    fsol2[0].plot(name, 'r-')
    fig.savefig(name+'.png', bbox_inches='tight')
show()
