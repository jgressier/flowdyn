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
#import flowdyn.solution.euler_riemann as sol

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

finit = rhs.fdata_fromprim([ 1., 0., 1. ]) # rho, u, p

fsol = solver.solve(finit, cfl, [endtime])

solver.show_perf()
mach_th = np.sqrt(((bcL['ptot']/bcR['p'])**(1./3.5)-1.)/.2)
error = np.sqrt(np.sum((fsol[-1].phydata('mach')-mach_th)**2)/nx)/mach_th 
print ("theoretical Mach : {:3.3f}\nerror : {:.2}".format(mach_th, error*100))

# Figure / Plot
for name in ['density', 'pressure', 'mach']:
	fig = figure(figsize=(10,8))
	fig.suptitle('flow in straight duct')
	ylabel(name)
	grid(linestyle='--', color='0.5')
	#finit.plot(name, 'k-.')
	fsol[0].plot(name, 'ko')
	#fig.savefig(name+'.png', bbox_inches='tight')
	show()
