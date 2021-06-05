# -*- coding: utf-8 -*-
"""
test integration methods
"""

#import cProfile
import matplotlib.pyplot as plt
#import numpy as np 
import aerokit.aero.unsteady1D as uq

from flowdyn.mesh  import unimesh
from flowdyn.field import *
from flowdyn.xnum  import *
from flowdyn.integration import rk3ssp
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc      as modeldisc
#import flowdyn.solution.euler_riemann as sol

meshsim  = unimesh(ncell=200,  length=10., x0=-4.)
meshref  = unimesh(ncell=1000, length=10., x0=-4.)

model1 = euler.model()
model2 = euler.model()

# upstream state with Mach number = normalized velocity
uL = uq.unsteady_state(rho=1.4, u=1.8, p=1.)
uR = uL._rankinehugoniot_from_ushock(ushock=0.)
#
bcL  = { 'type': 'dirichlet',  'prim':  [uL.rho, uL.u, uL.p] }
bcR  = { 'type': 'dirichlet',  'prim':  [uR.rho, uR.u, uR.p] }

def initdata(x, uL, uR):
    rho = uL.rho + (uR.rho-uL.rho)*(x>0.)
    u   = uL.u + (uR.u-uL.u)*(x>0.)
    p   = uL.p + (uR.p-uL.p)*(x>0.)
    return [rho, u, p]

xnum1 = muscl(minmod) # 
xnum2 = muscl(vanalbada) # 

rhs1 = modeldisc.fvm(model1, meshsim, xnum1, numflux='rusanov', bcL=bcL, bcR=bcR)
solver1 = rk3ssp(meshsim, rhs1)
rhs2 = modeldisc.fvm(model2, meshsim, xnum1, numflux='hlle', bcL=bcL, bcR=bcR)
solver2 = rk3ssp(meshsim, rhs2)

# computation
#
endtime = 2. # 2.8 for subsonic, 2.8 for supersonic
cfl     = 1.

finit = rhs1.fdata_fromprim(initdata(meshsim.centers(), uL, uR))

fsol1 = solver1.solve(finit, cfl, [endtime])
solver1.show_perf()
fsol2 = solver2.solve(finit, cfl, [endtime])
solver2.show_perf()

# Figure / Plot

for name in ['density', 'pressure', 'mach']:
    fig = plt.figure(figsize=(10,8))
    plt.ylabel(name)
    plt.grid(linestyle='--', color='0.5')
    fsol1[0].plot(name, 'b-')
    fsol2[0].plot(name, 'r-')
    fig.savefig(name+'.png', bbox_inches='tight')
plt.show()
