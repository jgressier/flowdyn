# -*- coding: utf-8 -*-
"""
test integration methods
"""

#import cProfile
import matplotlib.pyplot as plt
import numpy as np 

from flowdyn.mesh  import unimesh
#from flowdyn.field import *
from flowdyn.xnum  import *
from flowdyn.integration import rk3ssp
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc      as modeldisc

meshsim  = unimesh(ncell=100,  length=1.)
meshref  = unimesh(ncell=1000, length=1.)

model = euler.model()

bcL  = { 'type': 'per' }
bcR  = { 'type': 'per' }
xnum1 = { 'numflux': 'hllc', 'num': muscl(vanalbada) }
xnum2 = { 'numflux': 'centered', 'num': extrapol3() }#muscl(minmod)

rhs1 = modeldisc.fvm(model, meshsim, **xnum1, bcL=bcL, bcR=bcR)
solver1 = rk3ssp(meshsim, rhs1)
rhs2 = modeldisc.fvm(model, meshsim, **xnum2, bcL=bcL, bcR=bcR)
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
times = np.linspace(0, endtime, 2, endpoint=True)

fsol1 = solver1.solve(finit, cfl, times)
solver1.show_perf()
fsol2 = solver2.solve(finit, cfl, times)
solver2.show_perf()

#for s in fsol1, fsol2:
#    print(s[-1].time)
# Figure / Plot

for name in ['density']:
    fig = plt.figure(figsize=(10,8))
    plt.ylabel(name)
    plt.grid(linestyle='--', color='0.5')
    #finit.plot(name, 'k-.')
    finit.plot(name, 'k-')
    fsol1[-1].plot(name, 'b-')
    fsol2[-1].plot(name, 'r-')
    plt.legend(['initial', xnum1['numflux'], xnum2['numflux']], loc='upper left',prop={'size':10})  
    #fig.savefig(name+'.png', bbox_inches='tight')
plt.show()
