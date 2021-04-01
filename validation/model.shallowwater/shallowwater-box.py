import numpy as np
import matplotlib.pyplot as plt
import time
#
import flowdyn.mesh                as mesh
import flowdyn.modelphy.shallowwater as shallowwater
import flowdyn.modeldisc           as modeldisc
#import flowdyn.field               as field
from flowdyn.xnum        import *
from flowdyn.integration import *
plt.rcParams["animation.html"] = "jshtml"  # for matplotlib 2.1 and above, uses JavaScript

ncell = 300 # Définition de la mesh
ntime   = 50   # number of intermediate snapshots, only 1 is recommended
endtime = 1.
tsave   = np.linspace(0, endtime, num=ntime+1)
length = 3.
mesh = mesh.unimesh(ncell, length)
model = shallowwater.model(g=10.)

cfl = 1

# schéma linéaire: extrapol1(), extrapol2()=extrapolk(1), centered=extrapolk(-1), extrapol3=extrapol(1./3.) 
# schéma avec limitation non linéaire: muscl(LIMITER) avec LIMITER = minmod, vanalbada, vanleer, superbee
xmeth, xmethstr  = extrapol3(),'extrapol3'

# explicit, rk2, rk3ssp, rk4, implicit, trapezoidal=cranknicolson
tmeth, tmethstr = rk3ssp, 'rk3ssp'

# Numerical Flux : 'centered', 'rusanov', or 'HLL' are availaible. 
# Centered flux is easily shown to be unconditionnaly unstable
numflux = 'hll'

# Boundaries conditions : 'infinite' and 'sym' are available, unfortunaltely symmetric BC don't seem to work
bcL =  {'type':'infinite'}
bcR =  {'type':'infinite'}

# Initial conditions
def init_gouttedeau(mesh, max_height):
    return max_height*(-np.exp(-((mesh.centers()-1)/0.1)**2)) +.1   

h0_vect = init_gouttedeau(mesh,-0.01)

def init_rupturebarrage(mesh):
    h0_vect = np.zeros(len(mesh.centers()))+0.5
    for i in range(len(mesh.centers())):
        if mesh.centers()[i]<1 :
            h0_vect[i] = 1
    return h0_vect

#h0_vect = init_rupturebarrage(mesh)

u0_vect = .5 + np.zeros((ncell))
w_init = [h0_vect, u0_vect*h0_vect]
field0  = field.fdata(model, mesh, w_init)

# Define RHS
rhs = modeldisc.fvm(model, mesh, xmeth, numflux =numflux, bcL=bcL, bcR=bcR)
# Define the solver 
solver = tmeth(mesh,rhs)
# Solution
w_sol = solver.solve(field0,cfl,tsave)
# VERIFIED : w_sol contains solutions in conservative variables [h,q=h*u]
solver.show_perf()

fig2, (ax1,ax2)= plt.subplots(1,2,figsize=(15,7))
ax1.set_xlabel('$x$'), ax1.set_ylabel('$h(t,x)$'), ax1.set_title('Hauteur de fluide au temps t = ' +str(endtime)+ '\n (CFL '
          +str(cfl)+ ', flux : ' + numflux + ', ' + xmethstr +', '+ tmethstr+')')
ax1.set(xlim=(-0.1, mesh.centers()[-1]+0.1), ylim=(-0.0,.2))
ax1.grid(linestyle='--', color='0.5')
line1, = ax1.plot(mesh.centers(),w_sol[-1].data[0])
#line1, = ax1.plot(mesh.centers(),h0_vect)

ax2.set_xlabel('$x$'),ax2.set_ylabel('$u(t,x)$'), ax2.set_title('Vitesse du fluide au temps t = ' +str(endtime)+ '\n (CFL '
          +str(cfl)+ ', flux : ' + numflux + ', ' + xmethstr +', '+ tmethstr+')')
ax2.grid(linestyle='--', color='0.5')
line2, = ax2.plot(mesh.centers(),w_sol[-1].data[1]/w_sol[-1].data[0])

plt.show()
