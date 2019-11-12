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
#import pyfvm.solution.euler_riemann as sol

nx = 50

meshsim  = unimesh(ncell=nx,  length=10.)

model = euler.model()
bcL = { 'type': 'insub',  'ptot': 1.4, 'rttot': 1. }
bcR = { 'type': 'outsub', 'p': 1. }

rhs = modeldisc.fvm(model, meshsim, muscl(minmod), 
      bcL=bcL, bcR=bcR)
solver = rk2(meshsim, rhs)

# computation
#
endtime = 500.
cfl     = .5

finit = rhs.fdata(model.prim2cons([  1., 0., 1. ])) # rho, u, p

fsol = solver.solve(finit, cfl, [endtime])

solver.show_perf()
print "theoretical Mach", np.sqrt(((bcL['ptot']/bcR['p'])**(1./3.5)-1.)/.2)

# Figure / Plot
for name in ['density', 'pressure', 'mach']:
	fig = figure(figsize=(10,8))
	ylabel(name)
	grid(linestyle='--', color='0.5')
	#finit.plot(name, 'k-.')
	fsol[0].plot(name, 'ko')
	#fig.savefig(name+'.png', bbox_inches='tight')
	show()
