# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
#import numpy      as np
#import matplotlib as mp
from pylab import *
#from math import *

import pyfvm.mesh  as mesh
import pyfvm.model as model
from pyfvm.field import *
from pyfvm.xnum  import *
from pyfvm.integration import *

mesh400 = mesh.unimesh(ncell=400, length=1.)
mesh200 = mesh.unimesh(ncell=200, length=1.)
mesh100  = mesh.unimesh(ncell=100,  length=1.)

endtime = 1   # final physical time of simulation
ntime   = 1   # number of intermediate snapshots
tsave   = linspace(0, endtime, num=ntime+1) 

mymodel     = model.convmodel(1.)

# TODO : make init method for scafield 
# sinus packet
def init_sinpack(mesh):
    return sin(2*2*pi/mesh.length*mesh.centers())*(1+sign(-(mesh.centers()/mesh.length-.25)*(mesh.centers()/mesh.length-.75)))/2        
    
# periodic wave
def init_sinper(mesh):
    k = 2 # nombre d'onde
    return sin(2*k*pi/mesh.length*mesh.centers())
    
# square signal
def init_square(mesh):
    return (1+sign(-(mesh.centers()/mesh.length-.25)*(mesh.centers()/mesh.length-.75)))/2

initm   = init_sinper   # 
meshs   = [ mesh100 ]

cfls    = [ 0.5, 0.6, 1.4 ]
#cfls    = [ 0.5, 0.9, 1. , 1.1]

# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.) 
xmeths  = [ extrapol1() ]  

# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
tmeths  = [ rk4 ]

legends = [ '400 pts', '200 pts', '100 pts' ]

#boundary condition bc : type of boundary condition - "p"=periodic / "d"=Dirichlet
bc = 'p'

solvers = []
results = []
errors1 = []
errors2 = []
nbcalc     = max(len(cfls), len(tmeths), len(xmeths), len(meshs))

for i in range(nbcalc):
    field0 = scafield(mymodel, bc, (meshs*nbcalc)[i].ncell)
    field0.qdata[0] = initm((meshs*nbcalc)[i])
    solvers.append((tmeths*nbcalc)[i]((meshs*nbcalc)[i], (xmeths*nbcalc)[i]))
    start = time.clock()
    results.append(solvers[-1].solve(field0, (cfls*nbcalc)[i], tsave))
    print "> computation +legends[i] (",solvers[-1].nit,"it) :",time.clock()-start,"s"
    print "  cpu time :",time.clock()-start,"s"
    errors1.append(sum(abs(results[i][-1].qdata[0]-results[i][0].qdata[0]))/results[i][0].nelem)
    print "  L1 error :",errors1[i]
    errors2.append(sqrt(sum((results[i][-1].qdata[0]-results[i][0].qdata[0])**2)/results[i][0].nelem))
    print "  L2 error :",errors2[i]

# display and save results to png file
style=['o', 'x', 'D', '*', 'o', 'o']
fig=figure(1, figsize=(10,8))
plot(meshs[0].centers(), results[0][0].qdata[0], '-')
labels = ["initial condition"]
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plot((meshs*nbcalc)[i].centers(), results[i][t].qdata[0], style[i])
        labels.append(legends[i]+", t=%.1f"%results[i][t].time)
legend(labels, loc='upper left',prop={'size':10})  
fig.savefig('result.png', bbox_inches='tight')
show()
