# -*- coding: utf-8 -*-
"""
test integration methods
"""

#import cProfile
import matplotlib.pyplot as plt
import numpy as np 

import flowdyn.mesh2d as mesh2d
from flowdyn.field import *
from flowdyn.xnum  import extrapol2d1, extrapol2dk
import flowdyn.integration    as integ
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc      as modeldisc
#import flowdyn.solution.euler_riemann as sol

nx = 50
ny = 50

meshsim  = mesh2d.unimesh(nx, ny)

model = euler.euler2d()

#bcL = { 'type': 'insub',  'ptot': 1.4, 'rttot': 1. }
#bcR = { 'type': 'outsub', 'p': 1. }
bcper = { 'type': 'per' }
bcsym = { 'type': 'sym' }

rhs = modeldisc.fvm2d(model, meshsim, 
		num=extrapol2dk(1./3.),
#		num=extrapol2d1(),
		numflux='centered', 
		bclist={'left': bcper, 'right': bcper, 'top': bcper, 'bottom': bcper} )
#		bclist={'left': bcsym, 'right': bcsym, 'top': bcsym, 'bottom': bcsym} )
solver = integ.rk3ssp(meshsim, rhs)
# computation
#
endtime = 5.
cfl     = 1.

# initial functions
def fuv(x,y):
    #vmag = .01 ; k = 10.
    return euler.datavector(0.*x+.4, 0.*x+.2)
def fp(x,y): # gamma = 1.4
    return 0.*x+1.
def frho(x,y):
    return 1.4 * (1.+.5*np.exp(-((x-.5)**2+(y-.5)**2)/(.1)**2))

xc, yc = meshsim.centers()
finit = rhs.fdata_fromprim([ frho(xc, yc), fuv(xc, yc), fp(xc, yc) ]) # rho, (u,v), p

#cProfile.run("fsol = solver.solve(finit, cfl, [endtime])")
fsol = solver.solve(finit, cfl, [endtime])

solver.show_perf()

# Figure / Plot
vars = ['density', 'pressure', 'velocity_x', 'mach']
nvars = len(vars)
fig, ax = plt.subplots(ncols=nvars, figsize=(8*nvars-2,6))
fig.suptitle('density pulse: ')
for i, varname in enumerate(vars):
	ax[i].set_title(varname)
	#grid(linestyle='--', color='0.5')
	#finit.plot(name, 'k-.')
	cf = fsol[0].contourf(varname, axes=ax[i])
	fig.colorbar(cf, ax=ax[i])
	#fig.savefig(name+'.png', bbox_inches='tight')
plt.show()
