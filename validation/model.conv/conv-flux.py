# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
from pylab import *

import flowdyn.mesh  as mesh
import flowdyn.modelphy.convection as conv
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
from flowdyn.xnum  import *
from flowdyn.integration import *

mesh100 = mesh.unimesh(ncell=100, length=1.)
mesh50  = mesh.unimesh(ncell=50, length=1.)

mymodel = conv.model(1.)

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

initm   = init_sinper
meshs   = [ mesh100 ]

# Set of computations

endtime = 5
ntime   = 1
tsave   = linspace(0, endtime, num=ntime+1)
cfls    = [ .5 ]
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
xmeths  = [ extrapol1(), extrapol2(), centered(), extrapol3() ]
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
tmeths  = [ rk3ssp ]
legends = [ 'O1 upwind', 'O2 upwind', 'O2 centered', 'O3 extrapol' ]
#boundary condition bc : type of boundary condition - "p"=periodic / "d"=Dirichlet
bc = 'p'

solvers = []
results = []
nbcalc  = max(len(cfls), len(tmeths), len(xmeths), len(meshs))
for i in range(nbcalc):
    curmesh = (meshs*nbcalc)[i]
    finit = field.fdata(mymodel, curmesh, [ initm(curmesh) ] )
    rhs = modeldisc.fvm(mymodel, curmesh, (xmeths*nbcalc)[i])
    solvers.append((tmeths*nbcalc)[i](curmesh, rhs))
    results.append(solvers[-1].solve(finit, (cfls*nbcalc)[i], tsave)) #, flush="resfilename"))
    solvers[-1].show_perf()

# Figure

style = ['o', 'x', 'D', '*', '+', '>', '<', 'd']
fig = figure(1, figsize=(10,8))
grid(linestyle='--', color='0.5')
fig.suptitle('integration of various spatial scheme fluxes, CFL %.1f'%cfls[0], fontsize=12, y=0.93)
plot(meshs[0].centers(), results[0][0].data[0], '-')
labels = ["initial condition"]
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plot((meshs*nbcalc)[i].centers(), results[i][t].data[0], style[i])
        labels.append(legends[i]+", t=%.1f"%results[i][t].time)
legend(labels, loc='upper left',prop={'size':10})
fig.savefig('conv-flux.png', bbox_inches='tight')
show()

