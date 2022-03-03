# -*- coding: utf-8 -*-
"""
test integration methods
"""

#import cProfile
import matplotlib.pyplot as plt
import numpy as np 

from flowdyn.mesh import unimesh
from flowdyn.xnum import *
import flowdyn.integration as tnum
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc      as modeldisc

meshsim  = unimesh(ncell=200,  length=1.)
#meshref  = unimesh(ncell=1000, length=1.)

model = euler.model()

bcL  = { 'type': 'sym' } # not physical but can work
bcR  = { 'type': 'sym' } # for wall
xnum = muscl(vanalbada) ; flux = 'hllc'
#xnum = extrapol1() ; flux = 'centered'

rhs = modeldisc.fvm(model, meshsim, numflux=flux, num=xnum, bcL=bcL, bcR=bcR)
solver = tnum.lsrk26bb(meshsim, rhs)

# computation
#
nsol    = 100
endtime = .8
cfl     = .8

# initial functions
def fu(x):
    vmag = .01 #; k = 10.
    return vmag*np.exp(-500*(x-.2)**2) #*np.sin(2*np.pi*k*x)
def fp(x): # gamma = 1.4
    return (1. + .2*fu(x))**7.  # satisfies C- invariant to make only C+ wave
def frho(x):
    rhoratio = 10.
    return 1.4 * ( fp(x)**(1./1.4)*(x<.6) + rhoratio*(x>.6) )

xc    = meshsim.centers()
finit = rhs.fdata_fromprim([ frho(xc), fu(xc), fp(xc) ]) # rho, u, p

fsol = solver.solve(finit, cfl, np.linspace(0., endtime, nsol+1))
solver.show_perf()

# Figure / Plot

varname='pressure' # mach, pressure, entropy
ttime = [ fsol[i].time for i in range(nsol+1) ]
xx, xt = np.meshgrid(xc, ttime)
solgrid = [ fsol[i].phydata(varname) for i in range(nsol+1) ]
vmin, vmax = np.min(solgrid), np.max(solgrid)
#
# Figure / Plot of final results
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.set_ylabel(varname) ; ax1.set_ylim(vmin, vmax)
ax1.grid(linestyle='--', color='0.5')
finit.plot(varname, 'k-', axes=ax1)
line1, = fsol[-1].plot(varname, 'b-', axes=ax1)
ax2.set_ylabel('t') ; ax2.set_xlim(0., 1.)
#ax2.grid(linestyle='--', color='0.5')
#flood  = ax2.contour(xx, xt, solgrid, np.linspace(vmin, vmax, 50))
fsol.xtcontourf(varname, levels=51, axes=ax2)
line2, = ax2.plot([0., 10.], [ttime[-1], ttime[-1]], 'k--')
plt.show()
