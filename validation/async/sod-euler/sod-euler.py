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
from scipy.optimize import fsolve 
from pylab import *

from flowdyn.mesh  import *
from flowdyn.model import *
from flowdyn.field import *
from flowdyn.xnum  import *
from flowdyn.integration import *

mpl.rcParams['figure.dpi']      = 100
mpl.rcParams['savefig.dpi']     = 150
mpl.rcParams['text.usetex']     = True
mpl.rcParams['font.family']     = 'serif'

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{mathtools}  \usepackage{physics}')

cflmin    = .5
ncellmin  = 10
level     = 1
iteration = 2**(level-1)

cflmin   /= iteration
ncellmin *= iteration

nmesh    = nonunimesh(length=5., nclass=2, ncell0=ncellmin, periods=1) #fine,corase,fine
rmesh    = meshramzi(size=4, nclass = 3, length=5.)
umesh100 = unimesh(ncell=101, length=1.)

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

def classify(cfl, dx, data, bc):

    dt = mymodel.timestep(data, dx, cfl)

    dtmin = min(dt) #minimum Dt of all cells
    dtmax = max(dt) #calculated maximum Dt of all cells
    #print "dtmin",dtmin,"dtmax",dtmax

    nc = int(np.log2(dtmax/dtmin))  
    cell_class = np.full(len(dx),nc,dtype=int) #list of classes per cells initialize as maximum class

    for i in np.arange(len(dx)):
        cell_class[i] = nc-int(np.log2(dt[i]/dtmin))
    next = True
    while next == True:
        next = False
        #Forcing the same class for first and last cells to comply with periodic boundary conditions
        if bc == 'p':
            minclass = min(cell_class[0],cell_class[len(dx)-1])
            cell_class[0]         = minclass
            cell_class[len(dx)-1] = minclass

        for i in np.arange(len(dx)-1):

            iclass0 = cell_class[i]   #icv0 = i
            iclass1 = cell_class[i+1] #icv1 = i+1
            cldif   = iclass1-iclass0

            if abs(cldif) > 1:
                if cldif<0:
                    cell_class[i+1]=iclass0 - 1
                else:
                    cell_class[i]=iclass1 - 1
                next = True
        pass
    minclass = min(cell_class)
    maxclass = max(cell_class)
    nc = maxclass-minclass

    return cell_class

# Set of computations
endtime = 0.8
ntime   = 1
tsave   = linspace(0, endtime, num=ntime+1)
#type of asynchronous synchronisation sequence: 0 :=> [2 2 1 2 2 1 0] | 1 :=> [0 1 2 2 1 2 2] | 2 :=> [0 1 1 2 2 2 2]
asyncsq = 0  
# extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapolk(1./3.)
#xmeths  = [ extrapol1(), extrapol2(), centered(), extrapol3() ]
xmeths  = [ muscl() ]
# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
tmeths  = [ AsyncLSrk4 ]
#legends = [ 'O1 upwind', 'O2 upwind', 'O2 centered', 'O3 extrapol' ]
legends = [ 'O1 muscl' ]
#boundary condition bc : type of boundary condition - "p"=periodic / "d"=Dirichlet
bc       = 'd'
bcvalues = []
for i in range(mymodel.neq+1):
    bcvalues.append(np.zeros(2))

# Left Boundary

bcvalues[0][0] = 1.0      # density  rho
bcvalues[1][0] = 0.0      # velocity u       
bcvalues[2][0] = 2.5      # int. nrg e            
bcvalues[3][0] = 1.0      # pressure p            

# Right Boundary

bcvalues[0][1] = 0.125    # density  rho            
bcvalues[1][1] = 0.0      # velocity u            
bcvalues[2][1] = 2.0      # int. nrg e             
bcvalues[3][1] = 0.1      # pressure p            


gamma      = 1.4
meshs      = [ rmesh ]
initm      = initSod
exactPdata = exactSod(meshs[0],endtime)

# print len(meshs[0].dx()), meshs[0].dx()

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

cfls    = [cflmin, cflmin, cflmin]
tmeths  = [rk4, rk4, AsyncLSrk4]
legends = ['rk4', 'rk4', 'AsyncLSrk4']

# -----------------------------TEST lowstorage synchronous methods-----------------------------

# cfls    = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# tmeths  = [ rk4, sync_rk1, sync_rk22, sync_rk3ssp, sync_rk3lsw, sync_rk4]
# legends = [ 'rk4', 'sync_rk1', 'sync_rk22', 'sync_rk3ssp', 'sync_rk3lsw', 'sync_rk4']

#----------------------------------------------------------------------------------------------

solvers = []
results = []  
classes = []  

nbcalc  = max(len(cfls), len(tmeths), len(xmeths), len(meshs))
initial = list(np.empty(4))
test    = [[] for i in range(nbcalc)]


# First sync run with cflmin
i=0
field0 = scafield(mymodel, bc, (meshs*nbcalc)[i].ncell, bcvalues)
field0.qdata = initm((meshs*nbcalc)[i])                                  #initial solution
solvers.append((tmeths*nbcalc)[i]((meshs*nbcalc)[i], (xmeths*nbcalc)[i]))
start = time.clock()
results.append(solvers[-1].solve(field0, (cfls*nbcalc)[i], tsave)) #qdata
print "cpu time of '"+legends[i]+" computation (",solvers[-1].nit,"it) :",time.clock()-start,"s"
print "Final time of solution", results[i][1].time

classes = classify(cflmin, meshs[0].dx(), results[0][1].pdata, bc)
nc = max(classes)

cfls[1]  = cflmin/(2**nc)

# Rest of the runs for sync with cflmin/(2**nc) and async with cflmin
for i in range(1,nbcalc):
    field0 = scafield(mymodel, bc, (meshs*nbcalc)[i].ncell, bcvalues)
    field0.qdata = initm((meshs*nbcalc)[i])                                  #initial solution
    solvers.append((tmeths*nbcalc)[i]((meshs*nbcalc)[i], (xmeths*nbcalc)[i]))
    start = time.clock()
    results.append(solvers[-1].solve(field0, (cfls*nbcalc)[i], tsave)) #qdata

    print "cpu time of '"+legends[i]+" computation (",solvers[-1].nit,"it) :",time.clock()-start,"s"
    print "Final time of solution", results[i][1].time

classes = classify(cflmin, meshs[0].dx(), results[-1][1].pdata, bc)
# Initial solution
initial[0] = results[0][0].qdata[0]
initial[1] = results[0][0].qdata[1]/results[0][0].qdata[0]
initial[2] = (results[0][0].qdata[2]-0.5*results[0][0].qdata[1]**2/results[0][0].qdata[0])/results[0][0].qdata[0]
initial[3] = (gamma-1.0)*(results[0][0].qdata[2]-0.5*results[0][0].qdata[1]**2/results[0][0].qdata[0])

#Calling results[i][k] 
#i=0,nbcalc || which method 
#k=0,1      || 0:initial, 1:current
outdir = './'
if not os.path.exists(outdir):
    os.makedirs(outdir)
suffix =  tmeths[-1].__name__ + '_' + xmeths[0].__class__.__name__+'_asyncsq'+str(asyncsq)

#---------------------------------Plotting the characteristics of Euler's equation-------------------------
tend = 0.2                                         #total time for the figure
fig2=plt.figure(1, figsize=(10,8))
plt.ylabel(r"$t$")
plt.xlabel(r"$x$")
plt.grid()
plt.title("Characteristics of Euler's equation")
plt.axis([0, meshs[0].length, 0, endtime+tend])
for k in range(0,len(meshs[0].centers()),5):      #with a step not to plot all the characteristics
#plotting the characteristics for 0 < time < endtime
    xx = [meshs[0].centers()[k],exactPdata[0][k]]
    yy = [0.,endtime]
    lines = plt.plot(xx,yy)
    plt.setp(lines, color='black', linewidth=2.0)
#Plotting the continuation of the characteristics with dashed lines for endtime < time < endtime + tend   
    xx2 = [exactPdata[0][k],exactSod(meshs[0],endtime+tend)[0][k]]
    yy2 = [endtime,endtime+tend]
    linesd = plt.plot(xx2,yy2,'--')
    plt.setp(linesd, color='black', linewidth=2.0)    
fig2.savefig(outdir+'characteristics.pdf', bbox_inches='tight')
plt.show()
plt.close()
axisfontsize = 35
#-----------------------------------------------------------------------------------------
step = meshs[0].length/10.
style = ['o', 'x', 'D', '*', '+', '>', '<', 'd']



# *************************PLOTTING PRIME DATA AND ERRORS*********************************
symbol   = [r'\rho',r'u',r'e',r'p']
quantity = ['Density', 'Velocity', 'Internal energy', 'Pressure']
file     = ['density', 'velocity', 'internal_energy', 'pressure']
for eq in range(mymodel.neq+1):

    ref   = exactPdata[eq] #exact as reference data

    #-------------------------solving each Euler equation----------------------------------
    fig = figure(1, figsize=(10,8))
    grid(linestyle='--', color='0.5')
    fig.suptitle(quantity[eq]+' profile along the Sod shock-tube, CFL %.3f'%cfls[0], fontsize=12, y=0.93)
    plt.ylabel(r'$'+symbol[eq]+r'$',fontsize=axisfontsize)
    plt.xlabel(r"$x$",fontsize=axisfontsize)

    plot(meshs[0].centers(), initial[eq], '-.')
    # Exact solution
    plot(meshs[0].centers(), exactPdata[eq], 'g-')
    labels = ["initial condition","exact solution"+", t=%.1f"%results[0][len(tsave)-1].time]
    # Numerical solution
    for t in range(1,len(tsave)):
        for i in range(nbcalc):

            test[i].append(results[i][-1].qdata[0])
            test[i].append(results[i][-1].qdata[1]/results[i][-1].qdata[0])
            test[i].append((results[i][-1].qdata[2]-0.5*results[i][-1].qdata[1]**2/results[i][-1].qdata[0])/results[i][-1].qdata[0])
            test[i].append((gamma-1.0)*(results[i][-1].qdata[2]-0.5*results[i][-1].qdata[1]**2/results[i][-1].qdata[0]))

            plot((meshs*nbcalc)[i].centers(), test[i][eq], style[i])
            labels.append(legends[i]+", t=%.1f"%results[i][t].time)
    legend(labels, loc='lower left',prop={'size':10})
    fig.savefig(file[eq]+'.png', bbox_inches='tight')

    #-------------------------------------------Plotting (ref-test)/ref------------------------------------
    style =['-b','ob','-or', '-or', '-.or', ':or']
    mec   = ['b','b','r','r','r','r']
    markerfill = ['none','none','none','none','none','none','none']
    markersizes = [10 ,10, 10, 10, 10, 10, 10]
    fig, ax1 = plt.subplots(figsize=(10,8))
    plt.grid()
    #plt.title("Asynchronous vs Forward Euler for non-uniform mesh")
    #plt.xlim(0,meshs[0].length)
    #plt.xticks(arange(0,meshs[0].length+0.1,0.1))
    labels =[]
    for t in range(1,len(tsave)):    #first t gives us the initial condition
        for i in range(nbcalc):      #if first method is the reference method

            if not np.all(ref):
                error = np.abs(ref-test[i][eq])
                ylabel = r'$\abs{'+symbol[eq]+r'-'+symbol[eq]+r'_\text{ref}}$'
            else:
                error = np.abs((ref-test[i][eq])/ref)
                ylabel = r'$\abs{\frac{'+symbol[eq]+r'-'+symbol[eq]+r'_\text{ref}}{'+symbol[eq]+r'_\text{ref}}}$'

            ax1.plot((meshs*nbcalc)[i].centers(), error, style[i], markersize=markersizes[i], fillstyle=markerfill[i], markeredgecolor = mec[i])
    #        labels.append(legends[i]+", t=%.2f"%results[i][t].time+", CFL=%.2f"%cfls[i])
            labels.append(legends[i]+", CFL=%.2f"%cfls[i])

    ax1.set_xlabel(r'$x$',fontsize=axisfontsize)
    # ax1.set_ylabel(r'$\abs{U-U_\text{ref}}$', color='k',fontsize=axisfontsize)
    ax1.set_ylabel(ylabel, color='k',fontsize=axisfontsize)
    # ax1.set_ylim([min(uref)-0.02,max(uref)+0.02])        #in order to show the markers
    ax1.tick_params('both', colors='k',labelsize=18)
    #ax1.set_xlim([0,meshs[0].length])             #in order to show the squares
    ax1.xaxis.set_ticks(np.arange(0,meshs[0].length+0.1,step))

    #--------------------------------------------------------Plotting the classes------------------------------------
    x = (meshs*nbcalc)[0].xf                                             # the length of the nodes list is len(results)+1
    y = [float(i) for i in np.append(classes,classes[-1])]               #appends the class of the final cell in the end to match lengths
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
    ax2.tick_params('y', colors='k',labelsize=18)
    ax2.yaxis.set_ticks(np.arange(0, max(y)+1, 1))
    ax2.set_ylim([-0.02,max(y)+0.02])             #in order to show the cells
    ax2.set_xlim([0,meshs[0].length])             
    ax2.xaxis.set_ticks(np.arange(0,meshs[0].length+0.1,step))

    ax1.legend(labels, loc='best',prop={'size':16})  
    pdfname = 'error_rho_'+suffix+'.pdf'
    fig.savefig(outdir+pdfname, bbox_inches='tight')
    plt.show()

    #-----------------------------Mass calculation for all pdata---------------------------------
    print "********************************************************************"
    print "*                           %s                                " %quantity[eq]
    print "********************************************************************"
    mass0 = np.sum(results[0][0].pdata[eq]*meshs[0].dx())
    print "the mass integral of the initial condition is %.15f" %(mass0)
    error = []
    mass = []
    for i in range(nbcalc):           #for every time method
        u = results[i][1].pdata[eq]
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
        udif = u-ref
        #Calculating inf error
        Suw_inf = max(abs(udif))
        Surefw_inf = max(abs(ref))
        for c in range(len((meshs*nbcalc)[i].centers())):
            Sw  += dx[c]
            #Calculating L1 norm error
            Suw_L1 += abs(udif[c])*dx[c]
            Surefw_L1 += abs(ref[c])*dx[c]
            #Calculating L2 norm error
            Suw_L2 += (udif[c])**2*dx[c]
            Surefw_L2 += (ref[c])**2*dx[c]
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

    txtname = 'error_' + symbol[eq] + '_' + suffix +'.txt'
    error   = np.array(error[:])
    np.savetxt(outdir+txtname, error, delimiter='\t', header='L1, L2, inf', fmt='%12.8e') 

    txtname = 'mass_' + symbol[eq] + '_' + suffix + '.txt'
    np.savetxt(outdir+txtname, mass[3:]-mass0, delimiter='\t', header='mass', fmt='%12.8e')
    print