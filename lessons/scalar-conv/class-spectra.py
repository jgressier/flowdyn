# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
import numpy      as np
import matplotlib.pyplot as plt

import flowdyn.mesh  as mesh
import flowdyn.modelphy.convection as cmodel
from flowdyn.field import *
from flowdyn.xnum  import *
from flowdyn.integration import *
import flowdyn.modeldisc as modeldisc


mesh100  = mesh.unimesh(ncell=100,  length=1.)
mgmesh   = mesh.refinedmesh(ncell=100, length=1., ratio=2.)

mymodel     = cmodel.model(1.)

# TODO : make init method for scafield 
# sinus packet
def init_sinpack(mesh):
    return np.sin(2*2*np.pi/mesh.length*mesh.centers())*(1+np.sign(-(mesh.centers()/mesh.length-.25)*(mesh.centers()/mesh.length-.75)))/2        
    
# periodic wave
def init_sinper(mesh, k):
    #k = 2 # nombre d'onde
    return np.sin(2*k*np.pi/mesh.length*mesh.centers())
    
# square signal
def init_square(mesh):
    return (1+np.sign(-(mesh.centers()/mesh.length-.25)*(mesh.centers()/mesh.length-.75)))/2

def my_init(mesh):
    return init_sinper(mesh, k=2) 

# -----------------------------------------------------

meshs   = [ mesh100 ]
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.) 
xmeths   = [ extrapol1(), extrapol2(), extrapol3() ]
legends  = [ 'O1', 'O2', 'O3' ]
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson

#boundary condition bc : type of boundary condition - "p"=periodic / "d"=Dirichlet
bc = 'p'

results = []
labels  = []
nbcalc  = max(len(xmeths), len(meshs))

for i in range(nbcalc):
    thismesh = (meshs*nbcalc)[i]
    thisnum = (xmeths*nbcalc)[i]
    rhs = modeldisc.fvm(mymodel, thismesh, thisnum)
    field0 = rhs.fdata_fromprim([my_init(thismesh)])
    solver = implicit(thismesh, rhs)
    jac    = solver.calc_jacobian(field0)
    val, vec = np.linalg.eig(jac)
    results.append(val/thismesh.ncell)

# display and save results to png file
style=['o', 'x', 'D', '*', 'o', 'o']
fig=plt.figure(1, figsize=(10,8))
plt.clf()
for i in range(nbcalc):
    plt.scatter(results[i].real, results[i].imag, marker=style[i])
    labels.append(legends[i])
plt.legend(labels, loc='upper left',prop={'size':10})  
fig.savefig('res-spectra-xnum.png', bbox_inches='tight')
plt.show()

# -----------------------------------------------------

meshs   = [ mesh100, mgmesh ]
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.) 
xmeths   = [ extrapol3() ]
legends  = [ 'mesh100', 'mgmesh' ]

#boundary condition bc : type of boundary condition - "p"=periodic / "d"=Dirichlet
bc = 'p'

results = []
labels  = []
nbcalc  = max(len(xmeths), len(meshs))

for i in range(nbcalc):
    thismesh = (meshs*nbcalc)[i]
    thisnum = (xmeths*nbcalc)[i]
    rhs = modeldisc.fvm(mymodel, thismesh, thisnum)
    field0 = rhs.fdata_fromprim([my_init(thismesh)])
    solver = implicit(thismesh, rhs)
    jac    = solver.calc_jacobian(field0)
    val, vec = np.linalg.eig(jac)
    results.append(val/thismesh.ncell)

# display and save results to png file
style=['o', 'x', 'D', '*', 'o', 'o']
fig=plt.figure(2, figsize=(10,8))
plt.clf()
for i in range(nbcalc):
    plt.scatter(results[i].real, results[i].imag, marker=style[i])
    labels.append(legends[i])
plt.legend(labels, loc='upper left',prop={'size':10})  
fig.savefig('res-spectra-mesh.png', bbox_inches='tight')
plt.show()
