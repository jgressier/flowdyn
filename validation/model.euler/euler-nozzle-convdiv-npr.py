# -*- coding: utf-8 -*-
"""
test integration methods
"""

import cProfile, pstats, io

from matplotlib.pyplot import *
import numpy as np 

import flowdyn.mesh as mesh
from flowdyn.field import *
from flowdyn.xnum  import *
from flowdyn.integration import *
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc      as modeldisc
import flowdyn.solution.euler_nozzle as sol

NPR = 1.1

def S(x): # section law, throat is at x=5
    return 1.-.5*np.exp(-.5*(x-5.)**2)

model = euler.nozzle(sectionlaw=S)

meshsim  = mesh.unimesh(ncell=200,  length=10.)
meshref  = mesh.unimesh(ncell=1000, length=10.)

nozz = sol.nozzle(model, S(meshref.centers()), NPR=NPR)
fref = nozz.fdata(meshref)

bcL = { 'type': 'insub',  'ptot': NPR, 'rttot': 1. }
bcR = { 'type': 'outsub', 'p': 1. }

rhs = modeldisc.fvm(model, meshsim, muscl(vanleer), bcL=bcL, bcR=bcR)

solver = rk3ssp(meshsim, rhs)

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
sortby = 'cumulative' # SortKey.CUMULATIVE # only python >=3.7
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(25)
print(s.getvalue())

# Figure / Plot
for name in ['mach']:
	fig = figure(figsize=(10,8))
	ylabel(name)
	grid(linestyle='--', color='0.5')
	#finit.plot(name, 'k-.')
	fsol[0].plot(name, 'bo')
	fref.plot(name, 'k-')
	#fig.savefig(name+'.png', bbox_inches='tight')
	show()

