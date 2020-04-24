# -*- coding: utf-8 -*-
"""
test integration methods
"""

import time
from pylab import *

import pyfvm.mesh  as mesh
import pyfvm.modelphy.convection as conv
import pyfvm.modeldisc as modeldisc
import pyfvm.field as field
from pyfvm.xnum  import *
from pyfvm.integration import *

mlength = 1.0
mesh50  = mesh.unimesh(ncell=50, length=mlength)
mesh100 = mesh.unimesh(ncell=100, length=mlength)
mesh200 = mesh.unimesh(ncell=200, length=mlength)
mgmesh  = mesh.refinedmesh(ncell=75, length=mlength, ratio=2.0, nratioa=2)

endtime = 50.0
ntime   = 1
tsave   = linspace(0, endtime, num=ntime+1)

mymodel     = conv.model(1.0)

# TODO : make init method for scafield
# sinus packet
def init_sinpack_half(mesh):
    return sin(2*2*pi*mesh.centers())*(1+sign(-(mesh.centers()-.25)*(mesh.centers()-.75)))/2

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
meshs   = [ mesh100, mesh50, mgmesh ]
cfls    = [ .5 ]
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.)
xmeths  = [ extrapol3() ] ; xtl = "3rd order"
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
tmeths  = [ rk3ssp ] ; ttl = "RK3"
legends = [ '100 pts', '50 pts', 'refined' ]
#boundary condition bc : type of boundary condition - "p"=periodic / "d"=Dirichlet

solvers = []
results = []
nbcalc     = max(len(cfls), len(tmeths), len(xmeths), len(meshs))
for i in range(nbcalc):
    curmesh = (meshs*nbcalc)[i]
    finit = field.fdata(mymodel, curmesh, [ initm(curmesh) ] )
    rhs = modeldisc.fvm(mymodel, curmesh, (xmeths*nbcalc)[i])
    solvers.append((tmeths*nbcalc)[i](curmesh, rhs))
    results.append(solvers[-1].solve(finit, (cfls*nbcalc)[i], tsave)) #, flush="resfilename"))
    solvers[-1].show_perf()

style=['o', 'x', 'D', '*', '+', '>', '<', 'd']
fig=figure(1, figsize=(10,8))
clf()
fig.suptitle('integration on various grids, %s/%s , CFL %.2f'%(ttl,xtl,cfls[0]), fontsize=12, y=0.93)
plot(meshs[0].centers(), results[0][0].data[0], '-')
labels = ["initial condition"]
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plot((meshs*nbcalc)[i].centers(), results[i][t].data[0], style[i])
        labels.append(legends[i]+", t=%.1f"%results[i][t].time)
legend(labels, loc='upper left',prop={'size':10})
fig.savefig('conv-mesh.png', bbox_inches='tight')
show()
