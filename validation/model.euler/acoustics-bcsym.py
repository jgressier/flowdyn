
# -*- coding: utf-8 -*-
"""
test integration methods
"""
import numpy as np 
import matplotlib.pyplot as plt

import flowdyn.mesh as mesh
import flowdyn.xnum  as xnum
import flowdyn.integration as tnum
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc      as modeldisc

n = 500
h = 1./n
meshsim  = mesh.unimesh(ncell=n,  length=1.)

#meshref  = unimesh(ncell=1000, length=1.)

model = euler.model()

bcL  = { 'type': 'sym' } 
bcR  = { 'type': 'sym' } 
musclrus = { 'num': xnum.muscl(xnum.vanalbada), 'numflux':'rusanov' }
musclhllc = { 'num': xnum.muscl(xnum.vanalbada), 'numflux':'hllc' }
ctr2 = { 'num': xnum.extrapol1(), 'numflux':'centered' }
ctr4 = { 'num': xnum.extrapol3(), 'numflux':'centered' }

rhs = modeldisc.fvm(model, meshsim, **ctr2, bcL=bcL, bcR=bcR)
solver = tnum.lsrk25bb(meshsim, rhs)

# computation
#
gam = 1.4
M0 = 0.
a0 = 1.
p0 = 1.
xsig = .5
nsol    = 100
endtime = 2*(1.-xsig)/a0
cfl     = .5

# initial functions
Mmag = .00001 ; k = 20. ; isigx = 20.
togmu= 2./(gam-1.)
def fu(x):
    return a0*(M0+Mmag*np.exp(-(isigx*(x-xsig))**2))#*np.sin(2*np.pi*k*x))
def fp(x): # gamma = 1.4
    return p0*(1. + (fu(x)/a0-M0)/togmu)**(gam*togmu)  # satisfies C- invariant to make only C+ wave
def frho(x):
    rhoratio = 10.
    return gam * p0/a0**2 * ( fp(x)**(1./gam) )

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
from scipy.fft import fft, ifft
sp0 = h*fft(finit.phydata(varname))
sp1 = h*fft(fsol[-1].phydata(varname))
#
# Figure / Plot of final results
fig, ax = plt.subplots(2, 2, figsize=(12,8))
ax[0,0].set_ylabel(varname) ; ax[0,0].set_ylim(vmin, vmax)
ax[0,0].grid(linestyle='--', color='0.5')
finit.plot(varname, 'k-', axes=ax[0,0])
line1, = fsol[-1].plot(varname, 'b-', axes=ax[0,0])
ax[0,1].set_ylabel('t') ; ax[0,1].set_xlim(0., 1.)
#ax[0,1].grid(linestyle='--', color='0.5')
flood  = ax[0,1].contour(xx, xt, solgrid, np.linspace(vmin, vmax, 50))
line2, = ax[0,1].plot([0., 10.], [ttime[-1], ttime[-1]], 'k--')
#
n2 = 50
for sp, sty in [ (sp0, 'k.'), (sp1, 'b.')]:
    ax[1,0].plot(np.abs(sp[0:n2]), sty)
    ax[1,1].plot(np.angle(sp[0:n2]), sty)
ax[1,0].set_yscale('log')
ax[1,0].set_ylim(1e-12, 2*Mmag)
plt.show()
