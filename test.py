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

endtime = 1
ntime   = 1
tsave   = linspace(0, endtime, num=ntime+1) 

conv     = convmodel(1.)

# TODO : make init method for scafield 
# nombre d'onde
k = 1
field0 = scafield(conv, mesh1.ncell)
field0.qdata[0] = sin(2*k*pi/mesh1.length*mesh1.centers())   
#field0.qdata[0] = (1+sign(-(mesh1.centers()/mesh1.length-.25)*(mesh1.centers()/mesh1.length-.75)))/2
field0.qdata[0] = sin(2*2*pi/mesh1.length*mesh1.centers())*(1+sign(-(mesh1.centers()/mesh1.length-.25)*(mesh1.centers()/mesh1.length-.75)))/2 


meshs   = [ mesh1 ]
cfls    = [ 1., 5., 10.]
xmeths  = [ extrapol1 ]
tmeths  = [ time_implicit ]
legends = [ 'CFL 1', 'CFL 5', 'CFL 10']
#xmeth1   = extrapol1()
#xmeth2   = extrapol2()
#xmeth3   = extrapol3()
#expl_o1 = time_rk3ssp(mesh1, xmeth1)
#expl_o2 = time_implicit(mesh1, xmeth1)
#expl_o3 = time_rk3ssp(mesh1, xmeth3)
solvers = []
results = []   
nbcalc     = max(len(cfls), len(tmeths), len(xmeths), len(meshs))
for i in range(nbcalc):
    solvers.append((tmeths*nbcalc)[i]((meshs*nbcalc)[i], (xmeths*nbcalc)[i]()))
    start = time.clock()
    results.append(solvers[-1].solve(field0, (cfls*nbcalc)[i], tsave))
    print 'cpu time of "'+legends[i]+"' computation (",solvers[-1].nit," it) :",time.clock()-start

#results1 = expl_o1.solve(field0, cfl, tsave)
#results2 = expl_o2.solve(field0, cfl, tsave)
#results3 = expl_o3.solve(field0, cfl, tsave)

style=['-', 'o', 'x', 'D', '*', 'o', 'o']
fig=figure(1, figsize=(10,8))
plot(meshs[0].centers(), results[0][0].qdata[0], '-')
labels = ["initial condition"]
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plot((meshs*nbcalc)[i].centers(), results[i][t].qdata[0], style[i])
        labels.append(legends[i]+", t=%.1f"%results[i][t].time)
legend(labels, loc='upper left',prop={'size':10})  
fig.savefig('foo.png', bbox_inches='tight')
show()
