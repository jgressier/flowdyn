# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:56:32 2013

@author: j.gressier
"""

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

endtime = 5
ntime   = 1
tsave   = linspace(0, endtime, num=ntime+1)

cfl=0.5
conv     = convmodel(1.)
xmeth1   = extrapol1()
xmeth2   = extrapol2()
xmeth3   = extrapol3()
expl_o1 = time_rk3ssp(mesh1, xmeth1)
expl_o2 = time_rk3ssp(mesh1, xmeth2)
expl_o3 = time_rk3ssp(mesh1, xmeth3)

# TODO : make init method for scafield 
# nombre d'onde
k = 1
field0 = scafield(conv, mesh1.ncell)
field0.qdata[0] = sin(2*k*pi/mesh1.length*mesh1.centers())   
#field0.qdata[0] = (1+sign(-(mesh1.centers()/mesh1.length-.25)*(mesh1.centers()/mesh1.length-.75)))/2
field0.qdata[0] = sin(2*2*pi/mesh1.length*mesh1.centers())*(1+sign(-(mesh1.centers()/mesh1.length-.25)*(mesh1.centers()/mesh1.length-.75)))/2 

results1 = expl_o1.solve(field0, cfl, tsave)
results2 = expl_o2.solve(field0, cfl, tsave)
results3 = expl_o3.solve(field0, cfl, tsave)

print expl_o1.nit," iterations"

labels=[]
style=['-', 'o', 'x', 'D', '*', 'o', 'o']
figure(1, figsize=(10,8))
plot(mesh1.centers(), results1[0].qdata[0], '-')
labels.append("initial condition")
for i in range(1,len(results1)):
    plot(mesh1.centers(), results1[i].qdata[0], style[i])
    plot(mesh1.centers(), results2[i].qdata[0], style[i])
    plot(mesh1.centers(), results3[i].qdata[0], style[i])
    labels.append("O1, t=%.1f"%results1[i].time)
    labels.append("O2, t=%.1f"%results2[i].time)
    labels.append("O3, t=%.1f"%results3[i].time)
legend(labels, loc='upper left',prop={'size':10})  
show()
