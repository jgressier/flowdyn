# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
import cProfile
from pylab import plot
import numpy as np 

from pyfvm.mesh  import *
from pyfvm.field import *
from pyfvm.xnum  import *
from pyfvm.integration import *
import pyfvm.modelphy.euler as euler
import pyfvm.modeldisc      as modeldisc
import pyfvm.solution.euler_riemann as sol

meshsim  = unimesh(ncell=500,  length=10., x0=-4.)
meshref  = unimesh(ncell=1000, length=10., x0=-4.)

model = euler.model()
sod   = sol.Sod_subsonic(model)

bcL = { 'type': 'dirichlet',  'prim':  sod.bcL() }
bcR = { 'type': 'dirichlet',  'prim':  sod.bcR() }

rhs = modeldisc.fvm(model, meshsim, muscl(minmod), 
      bcL=bcL, bcR=bcR)
#      bcL={'type':'per'}, bcR={'type':'per'})
solver = rk3ssp(meshsim, rhs)

# computation
#
endtime = 2.8
cfl     = 1.

finit = sod.fdata(meshsim)

fsol = solver.solve(finit, cfl, [endtime])
solver.show_perf()

# Figure / Plot

fref = sod.fdata(meshref, endtime)

for name in ['density', 'pressure', 'mach']:
    fig = figure(figsize=(10,8))
    ylabel(name)
    grid(linestyle='--', color='0.5')
    #finit.plot(name, 'k-.')
    fref.plot(name, 'k-')
    fsol[0].plot(name, 'b-')
    fig.savefig(name+'.png', bbox_inches='tight')
show()
