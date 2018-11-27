# -*- coding: utf-8 -*-
"""
test integration methods
"""
import os
import time
import numpy as np
import subprocess
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
nlevels   = 4
time_method  = 'rk4'
space_method = 'muscl'

error_l1   = np.zeros((3,nlevels))
error_l2   = np.zeros((3,nlevels))
error_linf = np.zeros((3,nlevels))

ncell_arr  = np.zeros(nlevels)
level_arr  = np.zeros(nlevels)

for level in range(nlevels):

    ncellmin  = 4
    iteration = 2**(level-1)
    
    cflmin   /= iteration
    ncellmin *= iteration
    
    nmesh    = nonunimesh(length=5., nclass=2, ncell0=ncellmin, periods=1) #fine,corase,fine
    
    endtime = 6.
    ntime   = 1
    tsave   = linspace(0, endtime, num=ntime+1)
    
    mymodel = burgersinvmodel(dtmax=1.,dynamic=1)  #it takes as an argument a timestep dtmax which is the maximum timestep we need to capture the phenomena in the case study  
    
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
        return hat
    
    def init_step(mesh):
        step = np.zeros(len(mesh.centers()))
        ul   = 1.0
        ur   = 0.0
        xr   = 1.0
        x    = mesh.centers()
        for i in range(len(x)):
            if x[i] < xr:
                step[i] = ul
            elif xr <= x[i] <= 2.0:
                step[i] = 2.0-x[i] 
            elif x[i] > 2.0:
                step[i] = ur
        return step
    
    def exact(init,mesh,t):
        x  = mesh.centers() #original mesh
        u0 = init(mesh)     #depends on x0
        x1 = (x + u0*t)     #solution x for the characteristics
    
        alpha = 1.0
        ul    = 1.0
        ur    = 0.0
        xr    = 1.0
        tstar = 1.0/alpha 
        xstar = xr + ur * tstar
        s     = 0.5 * (ul+ur)
        u     = init(mesh)
        x    -= xr
    
        if t < tstar:
            for i in range(len(x)):
                if x[i] < t:
                    u[i] = ul
                elif x[i] >= ul*t and x[i] <= xr + ur*t:
                    u[i] = (ul-alpha*x[i])/(1.0-alpha*t)
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
        
    initm = init_step
    meshs = [ nmesh ]
    
    maxclass = 2   #the maximum number of classes
    boundary = 'd' #periodic: 'p' | dirichlet: 'd' |neumann: 'n'
    asyncsq  = 0   #type of asynchronous synchronisation sequence: 0 :=> [2 2 1 2 2 1 0] | 1 :=> [0 1 2 2 1 2 2] | 2 :=> [0 1 1 2 2 2 2]
    
    # extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.), muscl(limiter=minmod) 
    if space_method == 'extrapol1':
        xmeths  = [ extrapol1() ]
    elif space_method == 'extrapol2':
        xmeths  = [ extrapol2() ]
    elif space_method == 'centered':
        xmeths  = [ centered() ]
    elif space_method == 'extrapol3':
        xmeths  = [ extrapol2() ] 
    elif space_method == 'muscl':
        xmeths  = [ muscl() ]

    cfls    = [cflmin, cflmin/2**(maxclass-1), cflmin/2**(maxclass-1)]
           
    # -----------------------------TEST async rk1------------------------------------------------

    if time_method == 'rk1':
    
        tmeths  = [forwardeuler, forwardeuler, async_rk1]
        legends = ['rk1', 'rk1', 'async_rk1']
    
    # -----------------------------TEST async rk22------------------------------------------------
    
    elif time_method == 'rk2':
    
        tmeths  = [rk2, rk2, async_rk22]
        legends = ['rk2', 'rk2', 'async_rk22']
    
    # -----------------------------TEST async rk3ssp----------------------------------------------
    
    elif time_method == 'rk3ssp':
    
        tmeths  = [rk3ssp, rk3ssp, async_rk3ssp]
        legends = ['rk3ssp', 'rk3ssp', 'async_rk3ssp']
    
    # -----------------------------TEST async rk3lsw-----------------------------------------------
    
    elif time_method == 'rk3lsw':
    
        tmeths  = [sync_rk3lsw, sync_rk3lsw, async_rk3lsw]
        legends = ['sync_rk3lsw', 'sync_rk3lsw', 'async_rk3lsw']
    
    # -----------------------------TEST async rk4-----------------------------------------------
    elif time_method == 'rk4':
    
        tmeths  = [rk4, rk4, async_rk4]
        legends = ['rk4', 'rk4', 'async rk4']
    
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
    
    #Calling results[i][j][k] 
    #i=0,nbcalc || which method 
    #j=0,1      || 0:field, 1:class
    #k=0,1      || 0:initial, 1:current
    uref = exact(initm,meshs[0],endtime)[1] #exact as reference data
    #-----------------------------Error calculation for all time integration methods---------------------------------
    mass0 = np.sum(results[0][0][0].qdata[0]*meshs[0].dx())
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
        
        #Printing the mass
        #Printing the errors
        e_L1 = Suw_L1/(Sw*Surefw_L1)
        e_L2 = sqrt(Suw_L2/(Sw*Surefw_L2))
        e_Linf = Suw_inf/Surefw_inf

        error_l1[i,level]   = e_L1
        error_l2[i,level]   = e_L2        
        error_linf[i,level] = e_Linf 
        ncell_arr[level]    = len(nmesh.dx())
        level_arr[level]    = level+1
                
outdir = './'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# L1 ERROR ==============================================+++++======================================        

txtname_sync_clfmin_l1 = 'sync_cflmin' + str(asyncsq) +'_'+time_method+'_'+space_method+'_l1.txt'

matrix = np.array([ncell_arr,error_l1[0,:],level_arr]).transpose()
fmt = ['%5d', '%12.4e','%4d']
np.savetxt(outdir+txtname_sync_clfmin_l1, matrix, fmt, delimiter='\t', header = ' dof         l1_err   level', comments='')

txtname_sync_clfmax_l1 = 'sync_cflmax' + str(asyncsq) +'_'+time_method+'_'+space_method+'_l1.txt'
matrix = np.array([ncell_arr,error_l1[1,:],level_arr]).transpose()
fmt = ['%5d', '%12.4e','%4d']
np.savetxt(outdir+txtname_sync_clfmax_l1, matrix, fmt, delimiter='\t', header = ' dof         l1_err   level', comments='')

txtname_async_clfmin_l1 = 'async_cflmin' + str(asyncsq) +'_'+time_method+'_'+space_method+'_l1.txt'
matrix = np.array([ncell_arr,error_l1[2,:],level_arr]).transpose()
fmt = ['%5d', '%12.4e','%4d']
np.savetxt(outdir+txtname_async_clfmin_l1, matrix, fmt, delimiter='\t', header = ' dof         l1_err   level', comments='')

# L2 ERROR ==============================================+++++======================================        

txtname_sync_clfmin_l2 = 'sync_cflmin' + str(asyncsq) +'_'+time_method+'_'+space_method+'_l2.txt'

matrix = np.array([ncell_arr,error_l2[0,:],level_arr]).transpose()
fmt = ['%5d', '%12.4e','%4d']
np.savetxt(outdir+txtname_sync_clfmin_l2, matrix, fmt, delimiter='\t', header = ' dof         l2_err   level', comments='')

txtname_sync_clfmax_l2 = 'sync_cflmax' + str(asyncsq) +'_'+time_method+'_'+space_method+'_l2.txt'
matrix = np.array([ncell_arr,error_l2[1,:],level_arr]).transpose()
fmt = ['%5d', '%12.4e','%4d']
np.savetxt(outdir+txtname_sync_clfmax_l2, matrix, fmt, delimiter='\t', header = ' dof         l2_err   level', comments='')

txtname_async_clfmin_l2 = 'async_cflmin' + str(asyncsq) +'_'+time_method+'_'+space_method+'_l2.txt'
matrix = np.array([ncell_arr,error_l2[2,:],level_arr]).transpose()
fmt = ['%5d', '%12.4e','%4d']
np.savetxt(outdir+txtname_async_clfmin_l2, matrix, fmt, delimiter='\t', header = ' dof         l2_err   level', comments='')

# Linf ERROR ==============================================+++++======================================        

txtname_sync_clfmin_linf = 'sync_cflmin' + str(asyncsq) +'_'+time_method+'_'+space_method+'_linf.txt'

matrix = np.array([ncell_arr,error_linf[0,:],level_arr]).transpose()
fmt = ['%5d', '%12.4e','%4d']
np.savetxt(outdir+txtname_sync_clfmin_linf, matrix, fmt, delimiter='\t', header = ' dof         linf_err   level', comments='')

txtname_sync_clfmax_linf = 'sync_cflmax' + str(asyncsq) +'_'+time_method+'_'+space_method+'_linf.txt'
matrix = np.array([ncell_arr,error_linf[1,:],level_arr]).transpose()
fmt = ['%5d', '%12.4e','%4d']
np.savetxt(outdir+txtname_sync_clfmax_linf, matrix, fmt, delimiter='\t', header = ' dof         linf_err   level', comments='')

txtname_async_clfmin_linf = 'async_cflmin' + str(asyncsq) +'_'+time_method+'_'+space_method+'_linf.txt'
matrix = np.array([ncell_arr,error_linf[2,:],level_arr]).transpose()
fmt = ['%5d', '%12.4e','%4d']
np.savetxt(outdir+txtname_async_clfmin_linf, matrix, fmt, delimiter='\t', header = ' dof         linf_err   level', comments='')

# TEX FILE ==========================================================================================

texfile = '\
\
\documentclass[border=10pt]{standalone}\n\
\usepackage{verbatim}\n\
\usepackage{filecontents}\n\
\n\
\usepackage{pgfplots}\n\
\usepackage{pgfplotstable}\n\
\pgfplotsset{width=7cm,compat=1.8}\n\
\n\
\\begin{document}\n\
\n\
\\begin{tikzpicture}\n\
\\begin{loglogaxis}[\n\
    title='+space_method+'-'+time_method+',\n\
    xlabel={Number of cells},\n\
    ylabel={$L_1$ Error},\n\
    grid=major,\n\
    legend entries={sync CFL,async,sync CFL$/2$},\n\
    legend style = {font=\\footnotesize}\n\
]\n\
\\addplot[mark=*, color=blue] table {'+txtname_sync_clfmax_l1+'};\n\
\\addplot[mark=square, color=red!80!black] table {'+txtname_async_clfmin_l1+'};\n\
\\addplot[mark=square, color=blue] table {'+txtname_sync_clfmin_l1+'};\n\
\n\
%sync_l1 blue\n\
\\addplot[color=blue, dashed] table[\n\
x=dof,\n\
y={create col/linear regression={y=l1_err,\n\
		variance list={384,192,96,48,24,12}}}]\n\
{'+txtname_sync_clfmax_l1+'}\n\
% save two points on the regression line\n\
% for drawing the slope triangle\n\
coordinate [pos=0.65] (A)\n\
coordinate [pos=0.85]  (B)\n\
;\n\
% save the slope parameter:\n\
\\xdef\slope{\pgfplotstableregressiona}\n\
\n\
% draw the opposite and adjacent sides\n\
% of the triangle\n\
\draw (A) [color=blue] -|  (B)\n\
node [yshift=.7cm, xshift=.5cm,color=blue]\n\
{\pgfmathprintnumber{\slope}};\n\
\n\
%async0_l1 red\n\
\\addplot[color=red!80!black, dashed] table[\n\
x=dof,\n\
y={create col/linear regression={y=l1_err,\n\
		variance list={384,192,96,48,24,12}}}]\n\
{'+txtname_async_clfmin_l1+'}\n\
% save two points on the regression line\n\
% for drawing the slope triangle\n\
coordinate [pos=0.8] (A)\n\
coordinate [pos=0.55]  (B)\n\
;\n\
% save the slope parameter:\n\
\\xdef\slope{\pgfplotstableregressiona}\n\
\n\
% draw the opposite and adjacent sides\n\
% of the triangle\n\
\draw (A) [color=red!80!black] -|  (B)\n\
node [yshift=-.8cm, xshift=-.5cm,color=red!80!black]\n\
{\pgfmathprintnumber{\slope}};\n\
\n\
\end{loglogaxis}\n\
\end{tikzpicture}\n\
\n\
\\begin{tikzpicture}\n\
\\begin{loglogaxis}[\n\
    title='+space_method+'-'+time_method+',\n\
    xlabel={Number of cells},\n\
    ylabel={$L_2$ Error},\n\
    grid=major,\n\
    legend entries={sync CFL,async,sync CFL$/2$},\n\
    legend style = {font=\\footnotesize}\n\
]\n\
\\addplot[mark=*, color=blue] table {'+txtname_sync_clfmax_l2+'};\n\
\\addplot[mark=square, color=red!80!black] table {'+txtname_async_clfmin_l2+'};\n\
\\addplot[mark=square, color=blue] table {'+txtname_sync_clfmin_l2+'};\n\
\n\
%sync_l2 blue\n\
\\addplot[color=blue, dashed] table[\n\
x=dof,\n\
y={create col/linear regression={y=l2_err,\n\
		variance list={384,192,96,48,24,12}}}]\n\
{'+txtname_sync_clfmax_l2+'}\n\
% save two points on the regression line\n\
% for drawing the slope triangle\n\
coordinate [pos=0.65] (A)\n\
coordinate [pos=0.85]  (B)\n\
;\n\
% save the slope parameter:\n\
\\xdef\slope{\pgfplotstableregressiona}\n\
\n\
% draw the opposite and adjacent sides\n\
% of the triangle\n\
\draw (A) [color=blue] -|  (B)\n\
node [yshift=.7cm, xshift=.5cm,color=blue]\n\
{\pgfmathprintnumber{\slope}};\n\
\n\
%async0_l2 red\n\
\\addplot[color=red!80!black, dashed] table[\n\
x=dof,\n\
y={create col/linear regression={y=l2_err,\n\
		variance list={384,192,96,48,24,12}}}]\n\
{'+txtname_async_clfmin_l2+'}\n\
% save two points on the regression line\n\
% for drawing the slope triangle\n\
coordinate [pos=0.8] (A)\n\
coordinate [pos=0.55]  (B)\n\
;\n\
% save the slope parameter:\n\
\\xdef\slope{\pgfplotstableregressiona}\n\
\n\
% draw the opposite and adjacent sides\n\
% of the triangle\n\
\draw (A) [color=red!80!black] -|  (B)\n\
node [yshift=-.8cm, xshift=-.5cm,color=red!80!black]\n\
{\pgfmathprintnumber{\slope}};\n\
\n\
\end{loglogaxis}\n\
\end{tikzpicture}\n\
\n\
\n\
\\begin{tikzpicture}\n\
\\begin{loglogaxis}[\n\
    title='+space_method+'-'+time_method+',\n\
    xlabel={Number of cells},\n\
    ylabel={$L_\infty$ Error},\n\
    grid=major,\n\
    legend entries={sync CFL,async,sync CFL$/2$},\n\
    legend style = {font=\\footnotesize}\n\
]\n\
\\addplot[mark=*, color=blue] table {'+txtname_sync_clfmax_linf+'};\n\
\\addplot[mark=square, color=red!80!black] table {'+txtname_async_clfmin_linf+'};\n\
\\addplot[mark=square, color=blue] table {'+txtname_sync_clfmin_linf+'};\n\
\n\
%sync_linf blue\n\
\\addplot[color=blue, dashed] table[\n\
x=dof,\n\
y={create col/linear regression={y=linf_err,\n\
		variance list={384,192,96,48,24,12}}}]\n\
{'+txtname_sync_clfmax_linf+'}\n\
% save two points on the regression line\n\
% for drawing the slope triangle\n\
coordinate [pos=0.65] (A)\n\
coordinate [pos=0.85]  (B)\n\
;\n\
% save the slope parameter:\n\
\\xdef\slope{\pgfplotstableregressiona}\n\
\n\
% draw the opposite and adjacent sides\n\
% of the triangle\n\
\draw (A) [color=blue] -|  (B)\n\
node [yshift=.7cm, xshift=.5cm,color=blue]\n\
{\pgfmathprintnumber{\slope}};\n\
\n\
%async0_linf red\n\
\\addplot[color=red!80!black, dashed] table[\n\
x=dof,\n\
y={create col/linear regression={y=linf_err,\n\
		variance list={384,192,96,48,24,12}}}]\n\
{'+txtname_async_clfmin_linf+'}\n\
% save two points on the regression line\n\
% for drawing the slope triangle\n\
coordinate [pos=0.8] (A)\n\
coordinate [pos=0.55]  (B)\n\
;\n\
% save the slope parameter:\n\
\\xdef\slope{\pgfplotstableregressiona}\n\
\n\
% draw the opposite and adjacent sides\n\
% of the triangle\n\
\draw (A) [color=red!80!black] -|  (B)\n\
node [yshift=-.8cm, xshift=-.5cm,color=red!80!black]\n\
{\pgfmathprintnumber{\slope}};\n\
\n\
\end{loglogaxis}\n\
\end{tikzpicture}\n\
\n\
\n\
\end{document}'
texname = 'order_plot.tex'
pdfname = 'order_plot.pdf'
file = open(texname, 'w')
file.write(texfile)
file.close()

subprocess.call(['pdflatex', texname])
subprocess.call(['evince', pdfname])