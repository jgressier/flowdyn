# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
import cProfile
import matplotlib.pyplot as plt
import numpy as np 

import pyfvm.mesh2d as mesh2d
from pyfvm.field import *
from pyfvm.xnum  import *
from pyfvm.integration import *
import pyfvm.modelphy.euler as euler
import pyfvm.modeldisc      as modeldisc
#import pyfvm.solution.euler_riemann as sol

nx = 100
ny = 100
lx = 20
ly = 20
gam = 1.4
b=1.5
meshsim  = mesh2d.unimesh(nx, ny, lx, ly)

model = euler.euler2d()

bcper = { 'type': 'per' }
bcsym = { 'type': 'sym' }
xnum=extrapol2dk(k=1./3.)
#xnum=extrapol2d1()
rhs = modeldisc.fvm2d(model, meshsim, num=xnum, numflux='hlle', bclist={'left': bcper, 'right': bcper, 'top': bcper, 'bottom': bcper} )
solver = rk3ssp(meshsim, rhs)

# computation
#
endtime = 2. #10.
cfl     = 1.

# initial functions
def fuv(x,y):
    r=np.sqrt((x-lx/2)**2+(y-ly/2)**2)
    return euler.datavector(0.5-b/(2*np.pi)*np.exp(0.5*(1-r**2))*(y-ly/2), 0.5+b/(2*np.pi)*np.exp(0.5*(1-r**2))*(x-lx/2))
def fp(x,y): # gamma = 1.4
    return (frho(xc,yc))**gam
def frho(x,y):
    r=np.sqrt((x-lx/2)**2+(y-ly/2)**2)
    return (1-(((gam-1)*b**2)/(8*gam*np.pi**2))*np.exp(1-r**2))**(1/(gam-1))

xc, yc = meshsim.centers()

finit = rhs.fdata_fromprim([ frho(xc, yc), fuv(xc, yc), fp(xc, yc) ]) # rho, (u,v), p

#cProfile.run("fsol = solver.solve(finit, cfl, [endtime])")
fsol = solver.solve(finit, cfl, [endtime])

solver.show_perf()

# Figure / Plot
vars = ['pressure', 'density']#, 'mach']
nvars = len(vars)
fig, ax = plt.subplots(ncols=nvars, figsize=(10*nvars-2,6))
fig.suptitle('Isentropic Vortex: ')
for i, varname in enumerate(vars):
    ax[i].set_title(varname)
    #grid(linestyle='--', color='0.5')
    #finit.plot(name, 'k-.')
    cf = fsol[0].contourf(varname, axes=ax[i])
    fig.colorbar(cf, ax=ax[i])
    #fig.savefig(name+'.png', bbox_inches='tight')
plt.show()
