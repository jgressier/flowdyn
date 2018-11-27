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
ncellmin  = 2
level     = 3
iteration = 2**(level-1)

cflmin   /= iteration
ncellmin *= iteration

nmesh    = nonunimesh(length=1., nclass=2, ncell0=ncellmin, periods=1) #fine,corase,fine
rmesh    = meshramzi(size=3, nclass = 3)
umesh100 = unimesh(ncell=101, length=1.)

endtime = 1.
ntime   = 1
tsave   = linspace(0, endtime, num=ntime+1)
convvel = 1.

mymodel = convmodel(convvel)

print len(nmesh.dx())
print nmesh.dx()
# TODO : make init method for scafield 
# sinus packet
def init_sinpack(mesh):
    return sin(2*2*pi/mesh.length*mesh.centers())*(1+sign(-(mesh.centers()/mesh.length-.25)*(mesh.centers()/mesh.length-.75)))/2        
    
# periodic wave
def init_sinper(mesh):
    k = 1 # nombre d'onde
    return sin(2*k*pi/mesh.length*mesh.centers())
    
# square signal
def init_square(mesh):
    return (3+sign(-(mesh.centers()/mesh.length-.25)*(mesh.centers()/mesh.length-.75)))/2
    
def init_cos(mesh):
    k = 2 # nombre d'onde
    omega = k*pi/mesh.length
    return 1.-cos(omega*mesh.centers())

def init_hat(mesh):
    hat=zeros(len(mesh.centers()))
    xc = 0.5*mesh.length #center of hat
    y = 0.05              #height
    r = 0.2              #radius
    x1 = xc-r            #start of hat
    x2 = xc+r            #end of hat
    a1 = y/r
    b1 = -a1*x1
    a2 = -y/r
    b2 = -a2*x2
    k=0
    for i in mesh.centers():
        if x1 < i <= xc:
            hat[k]=a1*mesh.centers()[k]+b1
        elif xc < i < x2:
            hat[k]=a2*mesh.centers()[k]+b2
        k+=1
    return hat

def init_step(mesh):
    step = np.zeros(len(mesh.centers()))
    xc = 0.3*mesh.length #start of step
    y1 = 0.25            #height
    y2 = 0.
    r = 0.3              #radius
    x2 = xc+r            #end of step
    a2 = -y1/r
    b2 = -a2*x2
    k=0
    for i in mesh.centers():
        if i <= xc:
            step[k] = y1
        elif xc < i <= x2:
            step[k]=a2*i+b2
        else:
            step[k] = y2
        k+=1
    return step
   
def exact(init,mesh,t,a):
    x0 = mesh.centers()                           #original mesh
    u0 = init(mesh)                               #depends on x0
    x  = (x0 + a*t)%mesh.length                   #new solution x
    x1 = (x0 + a*t)                               #solution x for the characteristics
    return x1, x, u0

initm   = init_sinper
meshs   = [ nmesh ]

maxclass = 2 #the maximum number of classes
boundary = 'p' #periodic: 'p' | dirichlet: 'd' |neumann: 'n'
asyncsq  = 2   #type of asynchronous synchronisation sequence: 0 :=> [2 2 1 2 2 1 0] | 1 :=> [0 1 2 2 1 2 2] | 2 :=> [0 1 1 2 2 2 2]

# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.), muscl(limiter=minmod) 
xmeths  = [ extrapol3() ]
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson

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

cfls    = [cflmin, cflmin/2**(maxclass-1), cflmin/2**(maxclass-1)]
tmeths  = [rk4, rk4, async_rk4]
legends = ['rk4', 'rk4', 'async rk4']

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
    field0.qdata[0] = initm((meshs*nbcalc)[i])                                  #initial solution
    solvers.append((tmeths*nbcalc)[i]((meshs*nbcalc)[i], (xmeths*nbcalc)[i]))
    start = time.clock()
    results.append(solvers[-1].solve(field0, (cfls*nbcalc)[i], tsave))         #qdata and class
    print "cpu time of '"+legends[i]+" computation (",solvers[-1].nit,"it) :",time.clock()-start,"s"
    print "Final time of solution", results[i][0][1].time

#Calling results[i][j][k] 
#i=0,nbcalc || which method 
#j=0,1      || 0:field, 1:class
#k=0,1      || 0:initial, 1:current
outdir = './async_validation/convection/'
if not os.path.exists(outdir):
    os.makedirs(outdir)
#---------------------------------Plotting the characteristics of convection equation-------------------------
tend = 0.2                                         #total time for the figure
fig2=plt.figure(1, figsize=(10,8))
plt.ylabel(r"$t$")
plt.xlabel(r"$x$")
plt.grid()
plt.title("Characteristics of convection equation")
plt.axis([0, meshs[0].length, 0, endtime+tend])
for k in range(0,len(meshs[0].centers()),5):      #with a step not to plot all the characteristics
#plotting the characteristics for 0 < time < endtime
    xx = [meshs[0].centers()[k],exact(initm,meshs[0],endtime,convvel)[0][k]]
    yy = [0.,endtime]
    lines = plt.plot(xx,yy)
    plt.setp(lines, color='black', linewidth=2.0)
#Plotting the continuation of the characteristics with dashed lines for endtime < time < endtime + tend   
    xx2 = [exact(initm,meshs[0],endtime,convvel)[0][k],exact(initm,meshs[0],endtime+tend,convvel)[0][k]]
    yy2 = [endtime,endtime+tend]
    linesd = plt.plot(xx2,yy2,'--')
    plt.setp(linesd, color='black', linewidth=2.0)    
fig2.savefig(outdir+'characteristics.pdf', bbox_inches='tight')
plt.show()
plt.close()
axisfontsize = 22
#------------------------------------------------------Plotting u-------------------------------------------
style=['-ob','-r','or','--ok', '-ok', '-.ok', ':ok']
mec = ['b','r','r','k','k','k','k']
markersizes = [4 ,8, 8, 4, 4, 4, 4]
markerfill = ['full','none','none','none','none','none','none']
fig=plt.figure(2, figsize=(10,8))
plt.ylabel(r"$U$",fontsize=axisfontsize)
plt.xlabel(r"$x$",fontsize=axisfontsize)
plt.grid()
#plt.title("Asynchronous vs Forward Euler for non-uniform mesh")
plt.xlim(0,meshs[0].length)
plt.xticks(arange(0,meshs[0].length+0.1,0.1))
plot(meshs[0].centers(), results[0][0][0].qdata[0], '--k')                                      #plotting initial
plot(exact(initm,meshs[0],endtime,convvel)[1],exact(initm,meshs[0],endtime,convvel)[2],'-og')   #plotting exact
labels = ["initial condition", "exact solution"]
for t in range(1,len(tsave)):
    for i in range(nbcalc):
        plot((meshs*nbcalc)[i].centers(), results[i][0][t].qdata[0], style[i], markersize=markersizes[i], fillstyle=markerfill[i], markeredgecolor = mec[i])
#        labels.append(legends[i]+", t=%.2f"%results[i][0][t].time+", CFL=%.2f"%cfls[i])
        labels.append(legends[i]+", CFL=%.2f"%cfls[i])
legend(labels, loc='best',prop={'size':13})

suffix =  tmeths[-1].__name__ + '_' + xmeths[0].__class__.__name__+'_asyncsq'+str(asyncsq)
pdfname = 'u_conv_'+suffix+'.pdf' 

fig.savefig(outdir+pdfname, bbox_inches='tight')
plt.show()
#--------------------------------------------------------Plotting u-uref------------------------------------
# uref = results[0][0][0].qdata[0]                #init as reference data
uref = exact(initm,meshs[0],endtime,convvel)[2] #exact as reference data
style=['-b','ob','-or', '-or', '-.or', ':or']
mec = ['b','b','r','r','r','r']
markerfill = ['none','none','none','none','none','none','none']
markersizes = [10 ,10, 10, 10, 10, 10, 10]
fig, ax1 = plt.subplots(figsize=(10,8))
plt.ylabel(r"$U$",fontsize=axisfontsize)
plt.xlabel(r"$x$",fontsize=axisfontsize)
plt.grid()
#plt.title("Asynchronous vs Forward Euler for non-uniform mesh")
#plt.xlim(0,meshs[0].length)
#plt.xticks(arange(0,meshs[0].length+0.1,0.1))
labels =[]
for t in range(1,len(tsave)):      #first t gives us the initial condition
    for i in range(nbcalc):        #if first method is the reference method
        ax1.plot((meshs*nbcalc)[i].centers(), (results[i][0][t].qdata[0]-uref)/uref, style[i], markersize=markersizes[i], fillstyle=markerfill[i], markeredgecolor = mec[i])
#        labels.append(legends[i]+", t=%.2f"%results[i][0][t].time+", CFL=%.2f"%cfls[i])
        labels.append(legends[i]+", CFL=%.2f"%cfls[i])
ax1.set_xlabel(r'$x$',fontsize=axisfontsize)
ax1.set_ylabel(r'$\frac{U-U_{ref}}{U_{ref}}$', color='k',fontsize=axisfontsize)
ax1.tick_params('y', colors='k')
#ax1.set_xlim([0,meshs[0].length])             #in order to show the squares
#ax1.xaxis.set_ticks(np.arange(0,meshs[0].length+0.1,0.1))
#--------------------------------------------------------Plotting the classes------------------------------------
x = (meshs*nbcalc)[0].xf                                             # the length of the nodes list is len(results)+1
y = [float(i) for i in np.append(results[-1][1],results[-1][1][-1])]   #appends the class of the final cell in the end to match lengths
pos = np.where(np.abs(np.diff(y)) == 1)[0]                           #gets the indices where we have a change of class
x = np.insert(x, pos+1, np.nan)                                      #inserts in that index+1 a np.nan value that will prevent plotting it
y = np.insert(y, pos+1, np.nan)             
pos = np.where(np.isnan(y))[0]                                       #gets the indices of nan values
xadd = x[pos+1]                                                      #will add the xf of the index(nan)+1
yadd = y[pos-1]                                                      #will add the previous class in index(nan)-1 
k=0
for i in pos:
    x = np.insert(x, i+k, xadd[k])                                   
    y = np.insert(y, i+k, yadd[k])
    k += 1
ax2 = ax1.twinx()
ax2.plot(x,y,'-k|',markersize=7)
ax2.set_ylabel('classes', color='k',fontsize=20)
ax2.tick_params('y', colors='k')
ax2.yaxis.set_ticks(np.arange(0, max(y)+1, 1))
ax2.set_ylim([-0.02,max(y)+0.02])             #in order to show the squares
ax2.set_xlim([0,meshs[0].length])             
ax2.xaxis.set_ticks(np.arange(0,meshs[0].length+0.1,0.1))

ax1.legend(labels, loc='best',prop={'size':22})  
pdfname = 'error_conv_'+suffix+'.pdf'
fig.savefig(outdir+pdfname, bbox_inches='tight')
plt.show()
#-----------------------------Error calculation for all time integration methods---------------------------------
mass0 = np.sum(results[0][0][0].qdata[0]*meshs[0].dx())
print "the mass integral of the initial condition is %.15f" %(mass0)
error = []
mass = []
for i in range(nbcalc):           #for every time method
    u = results[i][0][1].qdata[0]
    dx = (meshs*nbcalc)[i].dx() 
    m = np.sum(u*dx)
    mass.append(m)
    Sw         = 0.
    Suw_L1     = 0.
    Surefw_L1  = 0.
    Suw_inf    = 0.
    Surefw_inf = 0.
    Suw_L2     = 0.
    Surefw_L2  = 0.
    udif = u-uref
#Calculating inf error
    Suw_inf = max(abs(udif))
    Surefw_inf = max(abs(uref))
    for c in range(len((meshs*nbcalc)[i].centers())):
        Sw  += dx[c]
#Calculating L1 norm error
        Suw_L1 += abs(udif[c])*dx[c]
        Surefw_L1 += abs(uref[c])*dx[c]
#Calculating L2 norm error
        Suw_L2 += (udif[c])**2*dx[c]
        Surefw_L2 += (uref[c])**2*dx[c]
    print "------------------------------%s------------------------------------" %(legends[i])
#Printing the mass
    print "the mass integral of method %s with CFL= %.2f is %.15f" %(legends[i], cfls[i], mass[i])
#Printing the errors
    error_L1 = Suw_L1/(Sw*Surefw_L1)
    print "the relative L1 error of method %s with CFL= %.2f is %.15f" %(legends[i], cfls[i], error_L1)
    error_L2 = sqrt(Suw_L2/(Sw*Surefw_L2))
    print "the relative L2 error of method %s with CFL= %.2f is %.15f" %(legends[i], cfls[i], error_L2)
    error_inf = Suw_inf/Surefw_inf
    print "the relative inf error of method %s with CFL= %.2f is %.15f" %(legends[i], cfls[i], error_inf)
    error.append([cfls[i],error_L1,error_L2 ,error_inf])
    
txtname = 'error_conv_' + suffix + '.txt'
error = np.array(error[:])
np.savetxt(outdir+txtname, error, delimiter='\t', header='L1, L2, inf', fmt='%.4e') 

txtname = 'mass_conv_' + suffix + '.txt'
np.savetxt(outdir+txtname, mass[3:], delimiter='\t', header='mass', fmt='%.4e') 