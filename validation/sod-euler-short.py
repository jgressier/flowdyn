# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
import cProfile
from pylab import *
import numpy as np 

from pyfvm.mesh  import *
from pyfvm.field import *
from pyfvm.xnum  import *
from pyfvm.integration import *
import pyfvm.modelphy.euler as euler
import pyfvm.modeldisc      as modeldisc
import pyfvm.solution.euler_riemann as sol

meshsim  = unimesh(ncell=100,  length=10., x0=-4.)
meshref  = unimesh(ncell=1000, length=10., x0=-4.)

model = euler.model()
sod   = sol.Sod_subsonic(model)

bcL = { 'type': 'dirichlet',  'prim':  sod.bcL() }
bcR = { 'type': 'dirichlet',  'prim':  sod.bcR() }

rhs = modeldisc.fvm(model, meshsim, muscl(minmod), 
      bcL=bcL, bcR=bcR)
#      bcL={'type':'per'}, bcR={'type':'per'})
solver = rk2(meshsim, rhs)

# computation
#
endtime = 2.8
cfl     = .5

finit = sod.fdata(meshsim)

start = time.clock()
fsol = solver.solve(finit, cfl, [endtime])
cputime = time.clock()-start

print "cpu time computation (",solver.nit,"it) :",cputime,"s"
print "  %.2f µs/cell/it"%(cputime*1.e6/solver.nit/meshsim.ncell)

# Figure / Plot

fref = sod.fdata(meshref, endtime)

for name in ['density', 'pressure', 'mach']:
    fig = figure(figsize=(10,8))
    ylabel(name)
    grid(linestyle='--', color='0.5')
    #finit.plot(name, 'k-.')
    fref.plot(name, 'k-')
    fsol[0].plot(name, 'ko')
    fig.savefig(name+'.png', bbox_inches='tight')
show()
