# -*- coding: utf-8 -*-
"""
test integration methods
"""
import os
import time
import numpy as np
import scipy.integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *
from scipy.optimize import fsolve

from pyfvm.mesh  import *
from pyfvm.model import *
from pyfvm.field import *
from pyfvm.xnum  import *
from pyfvm.integration import *

mpl.rcParams['figure.dpi']      = 100
mpl.rcParams['savefig.dpi']     = 150
mpl.rcParams['text.usetex']     = True
mpl.rcParams['font.family']     = 'serif'

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{mathtools}  \usepackage{physics}')

cflmin    = 1.
ncellmin  = 100
level     = 1
iteration = 2**(level-1)

cflmin   /= iteration
ncellmin *= iteration

nmesh    = nonunimesh(length=1., nclass=2, ncell0=ncellmin, periods=1) #fine,corase,fine
rmesh    = meshramzi(size=2, nclass = 1)
umesh100 = unimesh(ncell=101, length=1.)

endtime = 0.2
ntime   = 1
tsave   = linspace(0, endtime, num=ntime+1)

mymodel = eulermodel(dtmax=1.,dynamic=0)  #it takes as an argument a timestep dtmax which is the maximum timestep we need to capture the phenomena in the case study  

# TODO : make init method for scafield 
# sinus packet

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

def initSod(mesh):

    neq = 4 

    initEuler =  []
    for i in range(neq):
        initEuler.append(np.zeros(len(mesh.centers()))) #test use zeros instead

    gamma=1.4

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

    x    = mesh.centers()
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

initm = initSod
#meshs = [ umesh100 ]
meshs = [ nmesh ]
exactPdata = exactSod(meshs[0],endtime)


print len(meshs[0].dx()), meshs[0].dx()

maxclass = 2   #the maximum number of classes
boundary = 'd' #periodic: 'p' | dirichlet: 'd' |neumann: 'n'
asyncsq  = 0   #type of asynchronous synchronisation sequence: 0 :=> [2 2 1 2 2 1 0] | 1 :=> [0 1 2 2 1 2 2] | 2 :=> [0 1 1 2 2 2 2]

# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.), muscl(limiter=minmod) 
xmeths  = [ muscl() ]
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson

# -----------------------------TEST old RK1------------------------------------------------

# cfls    = [0.1, 0.2, 0.2/8., 0.2/8., 0.2/8., 0.2/8.]
# tmeths  = [ rk4, forwardeuler, ninterm, ninterm2, interm, interm2]
# legends = [ 'rk4', 'forwardeuler', 'ninterm', 'ninterm2', 'interm', 'interm2']

# -----------------------------TEST async rk1------------------------------------------------

# cfls    = [0.1, 0.2, 0.2/8., 0.2/8.]
# tmeths  = [ rk4, forwardeuler, forwardeuler, async_rk1]
# legends = [ 'rk4', 'forwardeuler', 'forwardeuler', 'async_rk1']

# -----------------------------TEST async rk22------------------------------------------------

# cfls    = [0.1, 0.2, 0.2/8., 0.2/8.]
# tmeths  = [ rk4, rk2, rk2, async_rk22]
# legends = [ 'rk4', 'rk2', 'rk2', 'async_rk22']

# -----------------------------TEST async rk3ssp----------------------------------------------

# cfls    = [0.1, 0.2, 0.2/8., 0.2/8.]
# tmeths  = [ rk4, rk3ssp, rk3ssp, async_rk3ssp]
# legends = [ 'rk4', 'rk3ssp', 'rk3ssp', 'async_rk3ssp']

# -----------------------------TEST async rk3lsw-----------------------------------------------

# cfls    = [0.1, 0.2, 0.2/8., 0.2/8.]
# tmeths  = [ rk4, sync_rk3lsw, sync_rk3lsw, async_rk3lsw]
# legends = [ 'rk4', 'sync_rk3lsw', 'sync_rk3lsw', 'async_rk3lsw']

# -----------------------------TEST async rk4-----------------------------------------------

#cfls    = [cflmin, cflmin/2**(maxclass-1), cflmin/2**(maxclass-1)]
#tmeths  = [rk4, rk4, async_rk4]
#legends = ['rk4', 'rk4', 'async rk4']

cfls    = [cflmin]
tmeths  = [rk4]
legends = ['rk4']

# -----------------------------TEST lowstorage synchronous methods-----------------------------

# cfls    = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# tmeths  = [ rk4, sync_rk1, sync_rk22, sync_rk3ssp, sync_rk3lsw, sync_rk4]
# legends = [ 'rk4', 'sync_rk1', 'sync_rk22', 'sync_rk3ssp', 'sync_rk3lsw', 'sync_rk4']

#----------------------------------------------------------------------------------------------

solvers = []
results = []  
classes = []  
nbcalc  = max(len(cfls), len(tmeths), len(xmeths), len(meshs))

for i in range(nbcalc):
    field0 = scafield(mymodel, maxclass, boundary, asyncsq, (meshs*nbcalc)[i].ncell)
    field0.qdata = initm((meshs*nbcalc)[i])                                  #initial solution
    solvers.append((tmeths*nbcalc)[i]((meshs*nbcalc)[i], (xmeths*nbcalc)[i]))
    start = time.clock()
    results.append(solvers[-1].solve(field0, (cfls*nbcalc)[i], tsave))         #qdata and class
    print "cpu time of '"+legends[i]+" computation (",solvers[-1].nit,"it) :",time.clock()-start,"s"
    print "Final time of solution", results[i][0][1].time

#Calling results[i][j][k] 
#i=0,nbcalc || which method 
#j=0,1      || 0:field, 1:class
#k=0,1      || 0:initial, 1:current
outdir = './async_validation/euler/'
if not os.path.exists(outdir):
    os.makedirs(outdir)
#------------------------------------------------------Plotting rho-------------------------------------------
axisfontsize = 35
style=['-ob','-r','or','--ok', '-ok', '-.ok', ':ok']
mec = ['b','r','r','k','k','k','k']
markersizes = [4 ,8, 8, 4, 4, 4, 4]
markerfill = ['full','none','none','none','none','none','none']
fig=plt.figure(2, figsize=(10,8))
plt.ylabel(r"$\rho$",fontsize=axisfontsize)
plt.xlabel(r"$x$",fontsize=axisfontsize)
plt.tick_params('both', colors='k',labelsize=18)
plt.grid()
#plt.title("Asynchronous vs Forward Euler for non-uniform mesh")
plt.xlim(0,meshs[0].length)
step = meshs[0].length/10.
plt.xticks(arange(0,meshs[0].length+0.1,step))
plot(meshs[0].centers(), results[0][0][0].qdata[0], '--k')         #plotting initial
# Exact solution
plot(meshs[0].centers(), exactPdata[0], '-')
labels = ["initial condition","exact condition"]


for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plot((meshs*nbcalc)[i].centers(), results[i][0][t].qdata[0], style[i], markersize=markersizes[i], fillstyle=markerfill[i], markeredgecolor = mec[i])
#        labels.append(legends[i]+", t=%.2f"%results[i][0][t].time+", CFL=%.2f"%cfls[i])
        labels.append(legends[0]+", CFL=%.2f"%cfls[i])
legend(labels, loc='best', prop={'size':16})
suffix =  tmeths[-1].__name__ + '_' + xmeths[0].__class__.__name__+'_asyncsq'+str(asyncsq)
pdfname = 'rho_euler_'+suffix+'.pdf'
fig.savefig(outdir+pdfname, bbox_inches='tight')
plt.show()

#------------------------------------------------------Plotting u-------------------------------------------
axisfontsize = 35
style=['-ob','-r','or','--ok', '-ok', '-.ok', ':ok']
mec = ['b','r','r','k','k','k','k']
markersizes = [4 ,8, 8, 4, 4, 4, 4]
markerfill = ['full','none','none','none','none','none','none']
fig=plt.figure(2, figsize=(10,8))
plt.ylabel(r"$U$",fontsize=axisfontsize)
plt.xlabel(r"$x$",fontsize=axisfontsize)
plt.tick_params('both', colors='k',labelsize=18)
plt.grid()
#plt.title("Asynchronous vs Forward Euler for non-uniform mesh")
plt.xlim(0,meshs[0].length)
step = meshs[0].length/10.
plt.xticks(arange(0,meshs[0].length+0.1,step))
plot(meshs[0].centers(), results[0][0][0].qdata[1], '--k')         #plotting initial
# Exact solution
plot(meshs[0].centers(), exactPdata[1], '-')
labels = ["initial condition","exact condition"]


for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plot((meshs*nbcalc)[i].centers(), results[i][0][t].qdata[1], style[i], markersize=markersizes[i], fillstyle=markerfill[i], markeredgecolor = mec[i])
#        labels.append(legends[i]+", t=%.2f"%results[i][0][t].time+", CFL=%.2f"%cfls[i])
        labels.append(legends[0]+", CFL=%.2f"%cfls[i])
legend(labels, loc='best', prop={'size':16})
suffix =  tmeths[-1].__name__ + '_' + xmeths[0].__class__.__name__+'_asyncsq'+str(asyncsq)
pdfname = 'u_euler_'+suffix+'.pdf'
fig.savefig(outdir+pdfname, bbox_inches='tight')
plt.show()


#------------------------------------------------------Plotting e-------------------------------------------
axisfontsize = 35
style=['-ob','-r','or','--ok', '-ok', '-.ok', ':ok']
mec = ['b','r','r','k','k','k','k']
markersizes = [4 ,8, 8, 4, 4, 4, 4]
markerfill = ['full','none','none','none','none','none','none']
fig=plt.figure(2, figsize=(10,8))
plt.ylabel(r"$e$",fontsize=axisfontsize)
plt.xlabel(r"$x$",fontsize=axisfontsize)
plt.tick_params('both', colors='k',labelsize=18)
plt.grid()
#plt.title("Asynchronous vs Forward Euler for non-uniform mesh")
plt.xlim(0,meshs[0].length)
step = meshs[0].length/10.
plt.xticks(arange(0,meshs[0].length+0.1,step))
plot(meshs[0].centers(), results[0][0][0].qdata[2], '--k')         #plotting initial
# Exact solution
plot(meshs[0].centers(), exactPdata[2], '-')
labels = ["initial condition","exact condition"]


for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plot((meshs*nbcalc)[i].centers(), results[i][0][t].qdata[2], style[i], markersize=markersizes[i], fillstyle=markerfill[i], markeredgecolor = mec[i])
#        labels.append(legends[i]+", t=%.2f"%results[i][0][t].time+", CFL=%.2f"%cfls[i])
        labels.append(legends[0]+", CFL=%.2f"%cfls[i])
legend(labels, loc='best', prop={'size':16})
suffix =  tmeths[-1].__name__ + '_' + xmeths[0].__class__.__name__+'_asyncsq'+str(asyncsq)
pdfname = 'e_euler_'+suffix+'.pdf'
fig.savefig(outdir+pdfname, bbox_inches='tight')
plt.show()

#------------------------------------------------------Plotting p-------------------------------------------
axisfontsize = 35
style=['-ob','-r','or','--ok', '-ok', '-.ok', ':ok']
mec = ['b','r','r','k','k','k','k']
markersizes = [4 ,8, 8, 4, 4, 4, 4]
markerfill = ['full','none','none','none','none','none','none']
fig=plt.figure(2, figsize=(10,8))
plt.ylabel(r"$p$",fontsize=axisfontsize)
plt.xlabel(r"$x$",fontsize=axisfontsize)
plt.tick_params('both', colors='k',labelsize=18)
plt.grid()
#plt.title("Asynchronous vs Forward Euler for non-uniform mesh")
plt.xlim(0,meshs[0].length)
step = meshs[0].length/10.
plt.xticks(arange(0,meshs[0].length+0.1,step))
plot(meshs[0].centers(), results[0][0][0].qdata[3], '--k')         #plotting initial
# Exact solution
plot(meshs[0].centers(), exactPdata[3], '-')
labels = ["initial condition","exact condition"]


for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plot((meshs*nbcalc)[i].centers(), results[i][0][t].qdata[3], style[i], markersize=markersizes[i], fillstyle=markerfill[i], markeredgecolor = mec[i])
#        labels.append(legends[i]+", t=%.2f"%results[i][0][t].time+", CFL=%.2f"%cfls[i])
        labels.append(legends[0]+", CFL=%.2f"%cfls[i])
legend(labels, loc='best', prop={'size':16})
suffix =  tmeths[-1].__name__ + '_' + xmeths[0].__class__.__name__+'_asyncsq'+str(asyncsq)
pdfname = 'p_euler_'+suffix+'.pdf'
fig.savefig(outdir+pdfname, bbox_inches='tight')
plt.show()
