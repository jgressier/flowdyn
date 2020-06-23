# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
import cProfile
from pylab import *
import numpy as np 

import pyfvm.mesh2d as mesh2d
from pyfvm.field import *
from pyfvm.xnum  import *
from pyfvm.integration import *
import pyfvm.modelphy.euler as euler
import pyfvm.modeldisc      as modeldisc
#import pyfvm.solution.euler_riemann as sol

nx = 4
ny = 4

meshsim  = mesh2d.unimesh(nx, ny)

model = euler.model()

#bcL = { 'type': 'insub',  'ptot': 1.4, 'rttot': 1. }
bcR = { 'type': 'outsub', 'p': 1. }

rhs = modeldisc.fvm2d(model, meshsim, num=None, bclist={} )
solver = rk3ssp(meshsim, rhs)

# computation
#
endtime = 1.
cfl     = 1.

# initial functions
def fuv(x,y):
    vmag = .01 ; k = 10.
    return euler.datavector(0.*x+.2, 0.*x)
def fp(x,y): # gamma = 1.4
    return 0.*x+1.
def frho(x,y):
    return 1.4 * (1+.2*np.exp(-((x-.5)**2+(y-.5)**2)/(.2)**2))

xc, yc = meshsim.centers()
print("x",xc)
print("u",fuv(xc, yc))
finit = rhs.fdata_fromprim([ frho(xc, yc), fuv(xc, yc), fp(xc, yc) ]) # rho, (u,v), p

fsol = solver.solve(finit, cfl, [endtime])

solver.show_perf()

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
