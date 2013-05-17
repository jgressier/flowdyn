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

endtime = 4.
ntime   = 2
tsave   = linspace(0, endtime, num=ntime+1)

conv     = convmodel(1.)
xmeth    = extrapol()
explicit = timeexplicit(mesh1, xmeth)

# nombre d'onde
k = 3
field0 = scafield(conv, mesh1.ncell)
#field0.qdata[0] = sin(2*k*pi/mesh1.length*mesh1.centers())   # TODO : make init method for scafield 
field0.qdata[0] = sign(mesh1.centers()/mesh1.length-.5)   # TODO : make init method for scafield 


results = explicit.solve(field0, 0.5, tsave)

labels=[]
style=['-', 'x', 'o', 'o', 'o', 'o']
figure(1, figsize=(10,8))
for i in range(len(results)):
    plot(mesh1.centers(), results[i].qdata[0], style[i])
    labels.append("t=%.1f"%results[i].time)
legend(labels, loc='upper left')  
show()
