# -*- coding: utf-8 -*-
"""
test integration methods
"""

import cProfile, pstats, io

import matplotlib.pyplot as plt
import numpy as np 

import flowdyn.mesh as mesh
from flowdyn.field import *
from flowdyn.xnum  import *
import flowdyn.integration as tnum
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc      as modeldisc
import flowdyn.solution.euler_nozzle as sol

NPR = 1.1

def S(x): # section law, throat is at x=5
    return 1.-.5*np.exp(-.5*(x-5.)**2)

model = euler.nozzle(gamma=1.35, sectionlaw=S)

meshsim  = mesh.unimesh(ncell=200,  length=10.)
meshref  = mesh.unimesh(ncell=1000, length=10.)

nozz = sol.nozzle(model, S(meshref.centers()), NPR=NPR)
fref = nozz.fdata(meshref)

bcL = { 'type': 'insub',  'ptot': NPR, 'rttot': 1. }
bcR = { 'type': 'outsub', 'p': 1. }

rhs = modeldisc.fvm(model, meshsim, muscl(vanleer), bcL=bcL, bcR=bcR)

monitors = {
    'residual':{ 'name':'pipo', 'frequency': 1 },
    'data_average':{ 'data': 'mach', 'name':'Mach average', 'frequency': 1 }
    }
solver = tnum.rk3ssp(meshsim, rhs, monitors=monitors)

# computation
#
endtime = 100.
cfl     = .8

finit = rhs.fdata_fromprim([  1., 0.1, 1. ]) # rho, u, p

pr = cProfile.Profile()
pr.enable()
fsol = solver.solve(finit, cfl, [endtime])
pr.disable()
solver.show_perf()

s = io.StringIO()
#sortby = 'cumulative' # SortKey.CUMULATIVE # only python >=3.7
sortby = 'time' # 
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(25)
print(s.getvalue())

# Figure / Plot
name = 'mach'
fig, ax = plt.subplots(1, 3, figsize=(22,8))
monres = monitors['residual']['output']
monavg = monitors['data_average']['output']
ax[0].set_ylabel(monavg.name)
ax[0].grid(linestyle='--', color='0.5')
ax[0].plot(monavg._it, monavg._value)
ax[1].set_ylabel(monres.name)
ax[1].grid(linestyle='--', color='0.5')
ax[1].semilogy(monres._it, monres._value)
ax[2].set_ylabel(name)
ax[2].grid(linestyle='--', color='0.5')
#finit.plot(name, 'k-.')
fsol[0].plot(name, 'bo', ax[2])
fref.plot(name, 'k-', ax[2])
#fig.savefig(name+'.png', bbox_inches='tight')
plt.show()

