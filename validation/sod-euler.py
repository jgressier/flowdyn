# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
from pylab import *

import numpy as np 
from scipy.optimize import fsolve 

from pyfvm.mesh  import *
from pyfvm.model import *
from pyfvm.field import *
from pyfvm.xnum  import *
from pyfvm.integration import *

mesh100 = unimesh(ncell=100, length=1.)
mesh50  = unimesh(ncell=50, length=1.)

mymodel = eulermodel()

# TODO : make init method for scafield
# Sod shocktube
def initSod(mesh):

    neq = 4 

    initEuler =  []
    for i in range(neq):
        initEuler.append(np.zeros(len(mesh.centers()))) #test use zeros instead

    rhoL = 1.0
    uL   = 0.0
    pL   = 1.0
    eL   = pL / ((gamma-1.0)*rhoL)
    EL   = eL + 0.5 * uL**2

    rhoR = 0.125
    uR   = 0.0
    pR   = 0.1
    eR   = pR / ((gamma-1.0)*rhoR)
    ER   = eR + 0.5 * uR**2

    x     = mesh.centers()
    xhalf = 0.5 * (x[0]+x[-1])

    for c in range(len(x)):
        if x[c] < xhalf:
            initEuler[0][c] = rhoL
            initEuler[1][c] = rhoL*uL
            initEuler[2][c] = rhoL*EL
            initEuler[3][c] = 0.0
        else:
            initEuler[0][c] = rhoR
            initEuler[1][c] = rhoR*uR
            initEuler[2][c] = rhoR*ER
            initEuler[3][c] = 0.0


    return initEuler

def exactSod(mesh,tf): # tf is the endtime

    neq = 4 

    exactEulerPdata =  []
    for i in range(neq):
        exactEulerPdata.append(np.zeros(len(mesh.centers()))) #test use zeros instead

    gamma = 1.4 
    gm1   = gamma - 1.0
    gp1   = gamma + 1.0
    g2    = 2.0*gamma 

    mu    = math.sqrt( gm1/gp1)
    beta  = gm1/g2

    # Initial conditions 
    rho1 = 1.0
    u1   = 0.0
    p1   = 1.0
    e1   = p1 / (gm1*rho1)
    E1   = e1 + 0.5 * u1**2

    rho5 = 0.125
    u5   = 0.0
    p5   = 0.1
    e5   = p5 / (gm1*rho5)
    E5   = e5 + 0.5 * u5**2

    #speed of sound 
    c1 = math.sqrt(gamma*p1/rho1)
    c5 = math.sqrt(gamma*p5/rho5)

    #location of the discontinuity at time t = 0 
    x  = mesh.centers()
    xi = 0.5 * (x[0]+x[-1])

    def f(p):
        z    = (p/p5-1.0) 
        fact = gm1 /g2 * (c5/c1) * z / math.sqrt(1.0+gp1 /g2 * z)
        fact = (1.0 - fact)**(g2/gm1)
        fp  = p1 * fact - p
        return fp

    p4 = fsolve(f, 0.5*(p1+p5))

    # resolve post shock density and velocity
    z    = (p4/p5-1.0) 
    gmfac1 = 0.5 *gm1/gamma
    gmfac2 = 0.5 *gp1/gamma

    fac = math.sqrt(1.0 + gmfac2 * z)

    u4   = c5 * z /(gamma * fac)
    rho4 = rho5 * (1.0 + gmfac2 * z) / (1.0 + gmfac1 * z) 

    # shock speed
    w = c5 * fac

    # compute the values at foot of the rarefaction wave
    p3   = p4
    u3   = u4
    rho3 = rho1 * (p3/p1)**(1.0/gamma)

    # compute the position of the waves 
    c3 =  math.sqrt(gamma*p3/rho3)

    xsh = xi +       w * tf # shock position
    xcd = xi +      u3 * tf # contact discontinuity position
    xft = xi + (u3-c3) * tf # rarefaction foot position
    xhd = xi -      c1 * tf # rarefaction head position

    for c in range(len(x)):
        if x[c] < xhd:
            e1   = p1 / ((gamma-1.0)*rho1)
            exactEulerPdata[0][c] = rho1
            exactEulerPdata[1][c] = u1
            exactEulerPdata[2][c] = e1
            exactEulerPdata[3][c] = p1
        elif x[c] < xft:
            u2   = 2.0 / gp1 * ( c1 + (x[c]-xi) / tf )
            fac  = 1.0 - 0.5 * gm1 * u2 / c1
            rho2 = rho1 * fac**(2.0/gm1)
            p2   = p1 * fac**(2.0*gamma / gm1)
            e2   = p2 / ((gamma-1.0)*rho2)
            exactEulerPdata[0][c] = rho2
            exactEulerPdata[1][c] = u2
            exactEulerPdata[2][c] = e2
            exactEulerPdata[3][c] = p2
        elif x[c] < xcd:
            e3   = p3 / ((gamma-1.0)*rho3)
            exactEulerPdata[0][c] = rho3
            exactEulerPdata[1][c] = u3
            exactEulerPdata[2][c] = e3
            exactEulerPdata[3][c] = p3
        elif x[c] < xsh:
            e4   = p4 / ((gamma-1.0)*rho4)
            exactEulerPdata[0][c] = rho4
            exactEulerPdata[1][c] = u4
            exactEulerPdata[2][c] = e4
            exactEulerPdata[3][c] = p4
        else:
            e5   = p5 / ((gamma-1.0)*rho5)
            exactEulerPdata[0][c] = rho5
            exactEulerPdata[1][c] = u5
            exactEulerPdata[2][c] = e5
            exactEulerPdata[3][c] = p5

    return exactEulerPdata

# Set of computations
endtime = 0.2
ntime   = 1
tsave   = linspace(0, endtime, num=ntime+1)
cfls    = [ 0.001 ]
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
#xmeths  = [ extrapol1(), extrapol2(), centered(), extrapol3() ]
xmeths  = [ extrapol1()]
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
tmeths  = [ rk3ssp ]
#legends = [ 'O1 upwind', 'O2 upwind', 'O2 centered', 'O3 extrapol' ]
legends = [ 'O1 upwind' ]
#boundary condition bc : type of boundary condition - "p"=periodic / "d"=Dirichlet
bc = 'd'

gamma      = 1.4
meshs      = [ mesh100 ]
initm      = initSod
exactPdata = exactSod(mesh100,endtime)

solvers = []
results = []
nbcalc  = max(len(cfls), len(tmeths), len(xmeths), len(meshs))
for i in range(nbcalc):
    field0 = scafield(mymodel, bc, (meshs*nbcalc)[i].ncell)
    field0.qdata = initm((meshs*nbcalc)[i])
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
rho0 = results[0][0].qdata[0]
plot(meshs[0].centers(), rho0, '-.')
# Exact solution
plot(meshs[0].centers(), exactPdata[0], '-')
labels = ["initial condition","exact condition"+", t=%.1f"%results[0][len(tsave)-1].time]
# Numerical solution
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        rho=results[i][t].qdata[0]
        plot((meshs*nbcalc)[i].centers(), rho, style[i])
        labels.append(legends[i]+", t=%.1f"%results[i][t].time)
legend(labels, loc='upper right',prop={'size':10})
fig.savefig('density.png', bbox_inches='tight')
#show()
#
# VELOCITY
#
fig = figure(2, figsize=(10,8))
grid(linestyle='--', color='0.5')
fig.suptitle('Velocity profile along the Sod shock-tube, CFL %.3f'%cfls[0], fontsize=12, y=0.93)
# Initial solution
u0 = results[0][0].qdata[1]/results[0][0].qdata[0]
plot(meshs[0].centers(), u0, '-.')             
# Exact solution
plot(meshs[0].centers(), exactPdata[1], '-')
labels = ["initial condition","exact condition"+", t=%.1f"%results[0][len(tsave)-1].time]
# Numerical solution
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        u = results[i][t].qdata[1]/results[i][t].qdata[0] 
        plot((meshs*nbcalc)[i].centers(), u, style[i])
        labels.append(legends[i]+", t=%.1f"%results[i][t].time)
legend(labels, loc='upper right',prop={'size':10})
fig.savefig('velocity.png', bbox_inches='tight')
#
# INTERNAL ENERGY
#
fig = figure(3, figsize=(10,8))
grid(linestyle='--', color='0.5')
fig.suptitle('Internal Energy profile along the Sod shock-tube, CFL %.3f'%cfls[0], fontsize=12, y=0.93)
# Initial solution
e0 = (results[0][0].qdata[2]-0.5*results[0][0].qdata[1]**2/results[0][0].qdata[0])/results[0][0].qdata[0]
plot(meshs[0].centers(), e0, '-.')
# Exact solution
plot(meshs[0].centers(), exactPdata[2], '-')
labels = ["initial condition","exact condition"+", t=%.1f"%results[0][len(tsave)-1].time]
# Numerical solution
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        e = (results[i][t].qdata[2]-0.5*results[i][t].qdata[1]**2/results[i][t].qdata[0])/results[i][t].qdata[0]
        plot((meshs*nbcalc)[i].centers(), e, style[i])
        labels.append(legends[i]+", t=%.1f"%results[i][t].time)
legend(labels, loc='upper right',prop={'size':10})
fig.savefig('internalenergy.png', bbox_inches='tight')
#
# PRESSURE
#
fig = figure(4, figsize=(10,8))
grid(linestyle='--', color='0.5')
fig.suptitle('Pressure profile along the Sod shock-tube, CFL %.3f'%cfls[0], fontsize=12, y=0.93)
# Initial solution
p0 = (gamma-1.0)*(results[0][0].qdata[2]-0.5*results[0][0].qdata[1]**2/results[0][0].qdata[0])
plot(meshs[0].centers(), p0, '-.')
# Exact solution
plot(meshs[0].centers(), exactPdata[3], '-') 
labels = ["initial condition","exact condition"+", t=%.1f"%results[0][len(tsave)-1].time]
# Numerical solution
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        p = (gamma-1.0)*(results[i][t].qdata[2]-0.5*results[i][t].qdata[1]**2/results[i][t].qdata[0])
        plot((meshs*nbcalc)[i].centers(), p, style[i])
        labels.append(legends[i]+", t=%.1f"%results[i][t].time)
legend(labels, loc='upper right',prop={'size':10})
fig.savefig('pressure.png', bbox_inches='tight')
                           
        
   


