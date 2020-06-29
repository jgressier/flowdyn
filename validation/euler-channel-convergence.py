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

solver1 = rk3ssp(meshsim, rhs)
solver2 = implicit(meshsim, rhs)

# computation
#
endtime = 200.
cfl1    = .8
cfl2    = 80.

finit = rhs.fdata(model.prim2cons([  1., 0.1, 1. ])) # rho, u, p

fsol1 = solver1.solve(finit, cfl1, [endtime])
solver1.show_perf()

fsol2 = solver2.solve(finit, cfl2, [endtime])
solver2.show_perf()

mach_th = np.sqrt(((bcL['ptot']/bcR['p'])**(1./3.5)-1.)/.2)
print("theoretical Mach", mach_th)

# Figure / Plot
for name in ['mach']:
	fig = figure(figsize=(10,8))
	ylabel(name)
	grid(linestyle='--', color='0.5')
	#finit.plot(name, 'k-.')
	fsol1[0].plot(name, 'ko')
	fsol2[0].plot(name, 'ro')
	#fig.savefig(name+'.png', bbox_inches='tight')
	show()
