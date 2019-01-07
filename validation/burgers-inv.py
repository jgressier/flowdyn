# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
from pylab import *

import numpy as np 
from scipy.optimize import fsolve 

from pyfvm.mesh  import *
import pyfvm.modelphy.burgers as burgers
from pyfvm.field import *
from pyfvm.xnum  import *
from pyfvm.integration import *

mesh50   = unimesh(ncell=50, length=5.)
mesh100  = unimesh(ncell=100, length=5.)
mesh1000 = unimesh(ncell=1000, length=5.)
nmesh    = nonunimesh(length=5., nclass=2, ncell0=10, periods=1) #fine,corase,fine
rmesh    = meshramzi(size=10, nclass = 3, length=5.)

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
tsave   = linspace(0, endtime, num=ntime+1)
cfls    = [ 0.5 ]
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
#xmeths  = [ extrapol1(), extrapol2(), centered(), extrapol3() ]
xmeths  = [ muscl() ]
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
tmeths  = [ LSrk4 ]
#legends = [ 'O1 upwind', 'O2 upwind', 'O2 centered', 'O3 extrapol' ]
legends = [ 'O1 muscl' ]
#boundary condition bc : type of boundary condition - "p"=periodic / "d"=Dirichlet / Neumann: 'n'
bc       = 'd'
bcvalues = []
for i in range(mymodel.neq+1):
    bcvalues.append(np.zeros(2))

# Left Boundary

bcvalues[0][0] = 2.0    #u        

# Right Boundary

bcvalues[0][1] = 1.0    #u                     

meshs      = [ mesh100 ]
initm      = init_step
exactPdata = exact_step(initm,meshs[0],endtime)

print len(meshs[0].dx()), meshs[0].dx()

solvers = []
results = []
nbcalc  = max(len(cfls), len(tmeths), len(xmeths), len(meshs))
for i in range(nbcalc):
    field0 = scafield(mymodel, bc, (meshs*nbcalc)[i].ncell, bcvalues)
    field0.qdata[0] = initm((meshs*nbcalc)[i])
    solvers.append((tmeths*nbcalc)[i]((meshs*nbcalc)[i], (xmeths*nbcalc)[i]))
    start = time.clock()
    results.append(solvers[-1].solve(field0, (cfls*nbcalc)[i], tsave))
    #print "cpu time of "+"%-11s"%(legends[i])+" computation (",solvers[-1].nit,"it) :",time.clock()-start,"s"

# Figure

style = ['o', 'x', 'D', '*', '+', '>', '<', 'd']
#
# Density
#
fig = figure(1, figsize=(10,8))
grid(linestyle='--', color='0.5')
fig.suptitle('Density profile along the Sod shock-tube, CFL %.3f'%cfls[0], fontsize=12, y=0.93)
# Initial solution
plot(meshs[0].centers(), results[0][0].qdata[0], '-')
# Exact solution
plot(meshs[0].centers(), exactPdata[1], '-')
labels = ["initial condition","exact solution"+", t=%.1f"%results[0][len(tsave)-1].time]
# Numerical solution
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plot((meshs*nbcalc)[i].centers(), results[i][t].qdata[0], style[i])
        labels.append(legends[i]+", t=%.1f"%results[i][t].time)
legend(labels, loc='lower left',prop={'size':10})
fig.savefig('density.png', bbox_inches='tight')
show()