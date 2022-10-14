# -*- coding: utf-8 -*-
"""
test integration methods
"""

import matplotlib.pyplot as plt
import numpy as np 

from flowdyn.mesh  import unimesh
from flowdyn.xnum  import *
import flowdyn.integration as tnum
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc      as modeldisc
import flowdyn.solution.euler_riemann as sol

meshsim  = unimesh(ncell=200,  length=1.)
meshref  = unimesh(ncell=1000, length=1.)

model = euler.model()

bcL  = { 'type': 'outsup' }
bcR  = { 'type': 'sym' }
xnum = muscl(vanalbada) ; flux = 'hllc'
#xnum = extrapol1() ; flux = 'centered'

rhs = modeldisc.fvm(model, meshsim, numflux=flux, num=xnum, bcL=bcL, bcR=bcR)
solver = tnum.rk3ssp(meshsim, rhs)

# computation
#
nsol    = 100
endtime = 1.6
cfl     = 1.

# initial functions
def fu(x):
    vmag = .001 ; k = 10.
    return vmag*np.exp(-200*(x-.2)**2) #*np.sin(2*np.pi*k*x)
def fp(x): # gamma = 1.4
    return (1. + .2*fu(x))**7.  # satisfies C- invariant to make only C+ wave
def frho(x):
    return 1.4 * fp(x)**(1./1.4)

xc    = meshsim.centers()
finit = rhs.fdata(model.prim2cons([ frho(xc), fu(xc), fp(xc) ])) # rho, u, p

fsol = solver.solve(finit, cfl, np.linspace(0., endtime, nsol+1))
solver.show_perf()

# Figure / Plot

# for name in ['pressure']:
#     fig = figure(figsize=(10,8))
#     ylabel(name)
#     grid(linestyle='--', color='0.5')
#     #finit.plot(name, 'k-.')
#     finit.plot(name, 'k-')
#     fsol[-1].plot(name, 'b-')
#     #legend(['initial', flux1, flux2], loc='upper left',prop={'size':10})  
#     fig.savefig(name+'.png', bbox_inches='tight')
# show()

icut=-1
varname='pressure' # mach, pressure, entropy
ttime = fsol[icut].time
stats = fsol.stats_solutions(varname)
# Figure / Plot of final results
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.set_ylabel(varname) ; ax1.set_ylim(stats['min'], stats['max'])
ax1.grid(linestyle='--', color='0.5')
finit.plot(varname, 'k-', axes=ax1)
line1, = fsol[-1].plot(varname, 'k-', axes=ax1)
ax2.set_ylabel('t') ; ax2.set_xlim(0., 1.)
#ax2.grid(linestyle='--', color='0.5')
fsol.xtcontour(varname, levels=np.linspace(stats['min'], stats['max'], 50), axes=ax2)
line2, = ax2.plot([0., 10.], [ttime, ttime], 'k--')
plt.show()
