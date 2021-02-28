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
length=10.
meshsim  = unimesh(ncell=nx,  length=length)

model = euler.model(source=[None, None, lambda x,q:.1])
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

vars = ['density', 'pressure', 'mach']
fig, axs = subplots(1, len(vars), figsize=(6*len(vars),6))
fig.suptitle('flow in straight duct')
# Figure / Plot
for name, ax in zip(vars, axs):
	ax.set_xlim(0., length)
	ax.set_ylabel(name)
	ax.grid(linestyle='--', color='0.5')
	#finit.plot(name, 'k-.')
	fsol[0].plot(name, 'o', axes=ax)
	#fig.savefig(name+'.png', bbox_inches='tight')
fig.tight_layout()
show()
