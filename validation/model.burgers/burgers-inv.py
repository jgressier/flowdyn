# -*- coding: utf-8 -*-
"""
test integration methods
"""

import matplotlib.pyplot as plt

import numpy as np 
from scipy.optimize import fsolve 

from pyfvm.mesh  import *
from pyfvm.field import *
from pyfvm.xnum  import *
import pyfvm.integration as integ
import pyfvm.modelphy.burgers as burgers
import pyfvm.modeldisc        as modeldisc

mlength = 5.
mesh50   = unimesh(ncell=50, length=mlength)
mesh100  = unimesh(ncell=100, length=mlength)
mesh1000 = unimesh(ncell=1000, length=mlength)
#nmesh    = nonunimesh(mlength, nclass=2, ncell0=10, periods=1) #fine,coarse,fine
#rmesh    = meshramzi(size=10, nclass = 3, length=mlength)

mymodel  = burgers.model()  #it takes as an argument a timestep dtmax which is the maximum timestep we need to capture the phenomena in the case study  

# TODO : make init method for scafield 
# sinus packet

def init_sinpack(mesh):
    return sin(2*2*pi/mesh.length*mesh.centers())*(1+sign(-(mesh.centers()/mesh.length-.25)*(mesh.centers()/mesh.length-.75)))/2        
    
# periodic wave
def init_cos(mesh):
    k = 2 # nombre d'onde
    omega = k*pi/mesh.length
    return 1.-cos(omega*mesh.centers())
    
def init_sin(mesh):
    k = 2 # nombre d'onde
    omega = k*pi/mesh.length
    return sin(omega*mesh.centers())
    
# square signal
def init_square(mesh):
    return (3+sign(-(mesh.centers()/mesh.length-.25)*(mesh.centers()/mesh.length-.75)))/2
    
def init_hat(mesh):
    hat = np.zeros(len(mesh.centers()))
    xc = 0.5*mesh.length #center of hat
    y = 0.04             #height
    r = 0.4              #radius
    x1 = xc-r            #start of hat
    x2 = xc+r            #end of hat
    a1 = y/r
    b1 = -a1*x1
    a2 = -y/r
    b2 = -a2*x2
    k=0
    for i in mesh.centers():
        if x1 < i <= xc:
            hat[k]=a1*i+b1
        elif xc < i < x2:
            hat[k]=a2*i+b2
        k+=1

    hat += 1.0

    return hat

def init_step(mesh):
    step = np.zeros(len(mesh.centers()))
    ul   = 2.0
    ur   = 1.0
    xr   = 1.0
    x    = mesh.centers()
    for i in range(len(x)):
        if x[i] < xr:
            step[i] = ul
        elif xr <= x[i] <= 2.0:
            step[i] = 3.0-x[i] 
        elif x[i] > 2.0:
            step[i] = ur
    return step

def exact_step(init,mesh,t):
    x  = mesh.centers() #original mesh
    u0 = init(mesh)     #depends on x0
    x1 = (x + u0*t)     #solution x for the characteristics

    alpha = 1.0
    ul    = 2.0
    ur    = 1.0
    xr    = 1.0
    tstar = 1.0/alpha 
    xstar = xr + ur * tstar
    s     = 0.5 * (ul+ur)
    u     = init(mesh)
    x    -= xr

    if t < tstar:
        for i in range(len(x)):
            if x[i] < ul*t:
                u[i] = ul
            elif x[i] >= ul*t and x[i] <= xr + ur*t:
                u[i] = (ul-alpha*x[i])/(ur-alpha*t)
            else:
                u[i] = ur
    else:
        shock_pos = xstar + s * (t-tstar)
        for i in range(len(x)):
            if x[i] < shock_pos:
                u[i] = ul
            else: 
                u[i] = ur

    return x1, u

# Set of computations
endtime = 2.
ntime   = 1
tsave   = np.linspace(0, endtime, num=ntime+1)
cfls    = [ 0.5 ]
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
#xmeths  = [ extrapol1(), extrapol2(), centered(), extrapol3() ]
xmeths  = [ muscl() ]
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
tmeths  = [ integ.rk4 ]
#legends = [ 'O1 upwind', 'O2 upwind', 'O2 centered', 'O3 extrapol' ]
legends = [ 'RK4 muscl' ]

meshs      = [ mesh100 ]
initm      = init_step
exactPdata = exact_step(initm, meshs[0], endtime)

solvers = []
results = []
nbcalc  = max(len(cfls), len(tmeths), len(xmeths), len(meshs))
for i in range(nbcalc):
    #f = field(mymodel, mesh100, np.zeros(100))
    thismesh = (meshs*nbcalc)[i]
    thisprim = [initm(thismesh)]
    thiscons = fdata(mymodel, thismesh, thisprim)
    bcL = { 'type': 'dirichlet', 'prim': thisprim[0]  }
    bcR = { 'type': 'dirichlet', 'prim': thisprim[-1] }
    rhs = modeldisc.fvm(mymodel, thismesh, (xmeths*nbcalc)[i], bcL=bcL, bcR=bcR)
    solvers.append((tmeths*nbcalc)[i](thismesh, rhs))
    results.append(solvers[-1].solve(thiscons, (cfls*nbcalc)[i], tsave))

# Figure

style = ['o', 'x', 'D', '*', '+', '>', '<', 'd']
#
# Density
#
fig = plt.figure(1, figsize=(10,8))
plt.grid(linestyle='--', color='0.5')
fig.suptitle('Burgers velocity profile, CFL %.3f'%cfls[0], fontsize=12, y=0.93)
# Initial solution
plt.plot(meshs[0].centers(), results[0][0].data[0], '-')
# Exact solution
plt.plot(meshs[0].centers(), exactPdata[1], '-')
labels = ["initial condition","exact solution"+", t=%.1f"%results[0][len(tsave)-1].time]
# Numerical solution
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plt.plot((meshs*nbcalc)[i].centers(), results[i][t].data[0], style[i])
        labels.append(legends[i]+", t=%.1f"%results[i][t].time)
fig.legend(labels, loc='lower left',prop={'size':10})
fig.savefig('Burgers.png', bbox_inches='tight')
plt.show()