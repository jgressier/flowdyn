# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
#import numpy      as np
#import matplotlib as mp
from pylab import *
#from math import *

from mesh  import *
from model import *
from field import *
from xnum  import *
from integration import *

mesh1 = unimesh(ncell=100, length=1.)

endtime = 10
ntime   = 1
tsave   = linspace(0, endtime, num=ntime+1) 

conv     = convmodel(1.)

# TODO : make init method for scafield 
# nombre d'onde
field0 = scafield(conv, mesh1.ncell)
# periodic wave 
k = 2
field0.qdata[0] = sin(2*k*pi/mesh1.length*mesh1.centers())
# square signal
#field0.qdata[0] = (1+sign(-(mesh1.centers()/mesh1.length-.25)*(mesh1.centers()/mesh1.length-.75)))/2
# sinus packet
field0.qdata[0] = sin(2*2*pi/mesh1.length*mesh1.centers())*(1+sign(-(mesh1.centers()/mesh1.length-.25)*(mesh1.centers()/mesh1.length-.75)))/2 


meshs   = [ mesh1 ]
cfls    = [ .9 ]
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=exgrapol(1./3.) 
xmeths  = [ extrapol3() ]  
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=crancknicholson
tmeths  = [ rk2, rk3ssp, implicit, trapezoidal ]
legends = [ 'O3-RK2', 'O3-RK3', 'O3-implicit', 'O3-Cranck-Nicholson',  ]

solvers = []
results = []   
nbcalc     = max(len(cfls), len(tmeths), len(xmeths), len(meshs))
for i in range(nbcalc):
    solvers.append((tmeths*nbcalc)[i]((meshs*nbcalc)[i], (xmeths*nbcalc)[i]))
    start = time.clock()
    results.append(solvers[-1].solve(field0, (cfls*nbcalc)[i], tsave))
    print "cpu time of '"+legends[i]+" computation (",solvers[-1].nit,"it) :",time.clock()-start,"s"

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
