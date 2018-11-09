# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
from pylab import *

import pyfvm.mesh  as mesh
import pyfvm.model as model
from pyfvm.field import *
from pyfvm.xnum  import *
from pyfvm.integration import *

mesh200 = mesh.unimesh(ncell=200, length=1.)
mesh100 = mesh.unimesh(ncell=100, length=1.)
mesh50  = mesh.unimesh(ncell=50,  length=1.)

endtime = .8   # final physical time of simulation
ntime   = 1    # number of intermediate snapshots
tsave   = linspace(0, endtime, num=ntime+1) 

mymodel     = model.convmodel(1.)

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

init   = init_sinper
mesh   = mesh100
cfl    = .9
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.) 
xmeth  = extrapol1()
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
tmeth  = explicit

#boundary condition bc : type of boundary condition - "p"=periodic / "d"=Dirichlet
bc = 'p'

# initialization
field0          = scafield(mymodel, bc, mesh.ncell)
field0.qdata[0] = init(mesh)

# solver integration
solver = tmeth(mesh, xmeth)
start = time.clock()
results = solver.solve(field0, cfl, tsave)
print "cpu time of computation (",solver.nit,"it) :",time.clock()-start,"s"

# display and save results to png file
style=['o', 'x', 'D', '*', 'o', 'o']
fig=figure(1, figsize=(10,8))
plot(mesh.centers(), results[0].qdata[0], '-')
labels = ["initial condition"]
for t in range(1,len(tsave)):
    plot(mesh.centers(), results[t].qdata[0], style[t-1])
    labels.append(", t=%.1f"%results[t].time)
legend(labels, loc='upper left',prop={'size':10})  
fig.savefig('result.png', bbox_inches='tight')
show()
