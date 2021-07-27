# -*- coding: utf-8 -*-
"""
test integration methods
"""

#import cProfile
from matplotlib.pyplot import figure, ylabel, grid, show
import numpy as np 

from flowdyn.mesh  import unimesh
#from flowdyn.field import *
from flowdyn.xnum  import *
import flowdyn.integration as tnum
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

solver1 = tnum.rk3ssp(meshsim, rhs)
solver2 = tnum.gear(meshsim, rhs)

# computation
#
endtime = 100.
cfl1    = .8
cfl2    = 20.

finit = rhs.fdata_fromprim([  1., 0.1, 1. ]) # rho, u, p

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
