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

meshsim  = unimesh(ncell=200,  length=1.)
meshref  = unimesh(ncell=1000, length=1.)

model = euler.model()

bcL  = { 'type': 'per' }
bcR  = { 'type': 'per' }
xnum1 = muscl(vanalbada)
flux1 = 'hllc'
xnum2 = extrapol1() #muscl(minmod)
flux2 = 'centered'

rhs1 = modeldisc.fvm(model, meshsim, numflux=flux1, num=xnum1, bcL=bcL, bcR=bcR)
solver1 = rk3ssp(meshsim, rhs1)
rhs2 = modeldisc.fvm(model, meshsim, numflux=flux2, num=xnum2, bcL=bcL, bcR=bcR)
solver2 = rk3ssp(meshsim, rhs2)

# computation
#
endtime = 50.
cfl     = 1.

# initial functions
def fu(x):
    return .3+0.*x  
def fp(x): 
    return 1.
def frho(x):
    return 1.4 * (1. + .2*np.exp(-(x-.5)**2/.01))

xc    = meshsim.centers()
finit = rhs1.fdata(model.prim2cons([ frho(xc), fu(xc), fp(xc) ])) # rho, u, p

fsol1 = solver1.solve(finit, cfl, [endtime])
solver1.show_perf()
fsol2 = solver2.solve(finit, cfl, [endtime])
solver2.show_perf()

# Figure / Plot

for name in ['density']:
    fig = figure(figsize=(10,8))
    ylabel(name)
    grid(linestyle='--', color='0.5')
    #finit.plot(name, 'k-.')
    finit.plot(name, 'k-')
    fsol1[0].plot(name, 'b-')
    fsol2[0].plot(name, 'r-')
    legend(['initial', flux1, flux2], loc='upper left',prop={'size':10})  
    #fig.savefig(name+'.png', bbox_inches='tight')
show()
