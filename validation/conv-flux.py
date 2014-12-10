# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
from pylab import *

from pyfvm.mesh  import *
from pyfvm.model import *
from pyfvm.field import *
from pyfvm.xnum  import *
from pyfvm.integration import *

mesh100 = unimesh(ncell=100, length=1.)
mesh50  = unimesh(ncell=50, length=1.)

endtime = 5
ntime   = 1
tsave   = linspace(0, endtime, num=ntime+1) 

mymodel     = convmodel(1.)

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
cfls    = [ .5 ]
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.) 
xmeths  = [ extrapol1(), extrapol2(), centered(), extrapol3()  ]  
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=crancknicholson
tmeths  = [ rk3ssp ]
legends = [ 'O1 upwind', 'O2 upwind', 'O2 centered', 'O3'  ]

solvers = []
results = []   
nbcalc     = max(len(cfls), len(tmeths), len(xmeths), len(meshs))
for i in range(nbcalc):
    field0 = scafield(mymodel, (meshs*nbcalc)[i].ncell)
    field0.qdata[0] = initm((meshs*nbcalc)[i])
    solvers.append((tmeths*nbcalc)[i]((meshs*nbcalc)[i], (xmeths*nbcalc)[i]))
    start = time.clock()
    results.append(solvers[-1].solve(field0, (cfls*nbcalc)[i], tsave))
    print "cpu time of '"+legends[i]+" computation (",solvers[-1].nit,"it) :",time.clock()-start,"s"

style=['o', 'x', 'D', '*', '+', '>', '<', 'd']
fig=figure(1, figsize=(10,8))
fig.suptitle('integration of various flux, CFL %.1f'%cfls[0], fontsize=12, y=0.93)
plot(meshs[0].centers(), results[0][0].qdata[0], '-')
labels = ["initial condition"]
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plot((meshs*nbcalc)[i].centers(), results[i][t].qdata[0], style[i])
        labels.append(legends[i]+", t=%.1f"%results[i][t].time)
legend(labels, loc='upper left',prop={'size':10})  
fig.savefig('conv-flux.png', bbox_inches='tight')
#show()
