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

mesh100  = mesh.unimesh(ncell=100,  length=1.)
mgmesh   = mesh.refinedmesh(ncell=100, length=1., ratio=2.)

mymodel     = model.convmodel(1.)

# TODO : make init method for scafield 
# sinus packet
def init_sinpack(mesh):
    return sin(2*2*pi/mesh.length*mesh.centers())*(1+sign(-(mesh.centers()/mesh.length-.25)*(mesh.centers()/mesh.length-.75)))/2        
    
# periodic wave
def init_sinper(mesh, k):
    #k = 2 # nombre d'onde
    return sin(2*k*pi/mesh.length*mesh.centers())
    
# square signal
def init_square(mesh):
    return (1+sign(-(mesh.centers()/mesh.length-.25)*(mesh.centers()/mesh.length-.75)))/2

def my_init(mesh):
    return init_sinper(mesh, k=2) 

# -----------------------------------------------------

meshs   = [ mesh100 ]
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.) 
xmeths   = [ extrapol1(), extrapol2(), extrapol3() ]
legends  = [ 'O1', 'O2', 'O3' ]
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson

results = []
labels  = []
nbcalc  = max(len(xmeths), len(meshs))

for i in range(nbcalc):
    lmesh = (meshs*nbcalc)[i]
    field0 = scafield(mymodel, lmesh.ncell)
    field0.qdata[0] = my_init(lmesh)
    solver = implicit(lmesh, (xmeths*nbcalc)[i])
    jac    = solver.calc_jacobian(numfield(field0))
    val, vec = eig(jac)
    results.append(val/lmesh.ncell)

# display and save results to png file
style=['o', 'x', 'D', '*', 'o', 'o']
fig=figure(1, figsize=(10,8))
clf()
for i in range(nbcalc):
    scatter(results[i].real, results[i].imag, marker=style[i])
    labels.append(legends[i])
legend(labels, loc='upper left',prop={'size':10})  
fig.savefig('res-spectra-xnum.png', bbox_inches='tight')
show()

# -----------------------------------------------------

meshs   = [ mesh100, mgmesh ]
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.) 
xmeths   = [ extrapol3() ]
legends  = [ 'mesh100', 'mgmesh' ]

results = []
labels  = []
nbcalc  = max(len(xmeths), len(meshs))

for i in range(nbcalc):
    lmesh = (meshs*nbcalc)[i]
    field0 = scafield(mymodel, lmesh.ncell)
    field0.qdata[0] = my_init(lmesh)
    solver = implicit(lmesh, (xmeths*nbcalc)[i])
    jac    = solver.calc_jacobian(numfield(field0))
    val, vec = eig(jac)
    results.append(val/lmesh.ncell)

# display and save results to png file
style=['o', 'x', 'D', '*', 'o', 'o']
fig=figure(2, figsize=(10,8))
clf()
for i in range(nbcalc):
    scatter(results[i].real, results[i].imag, marker=style[i])
    labels.append(legends[i])
legend(labels, loc='upper left',prop={'size':10})  
fig.savefig('res-spectra-mesh.png', bbox_inches='tight')
show()
