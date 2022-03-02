# -*- coding: utf-8 -*-
"""
Nozzle initialization, computation and restart
"""

import matplotlib.pyplot as plt
import numpy as np
# library for theoretical computation of flow regime
import aerokit.aero.MassFlow as mf
import aerokit.aero.ShockWave as sw
import aerokit.aero.Isentropic as Is
# Euler/Nozzle simulations
import flowdyn.mesh as mesh
from flowdyn.xnum import *
from flowdyn.integration import *
import flowdyn.modelphy.euler as euler
import flowdyn.modeldisc as modeldisc
import flowdyn.solution.euler_nozzle as sol

gam = 1.4
bctype = "outsub_rh"
ncell = 100
nit_super = 1000
nit_tot = 10000

# expected Mach number at exit when supersonic ; defines As/Ac ratio
Msup = 1.8
AsAc = mf.Sigma_Mach(Msup, gam)
Msub = mf.MachSub_Sigma(AsAc, gam)
NPRsup = Is.PtPs_Mach(Msup, gam)
NPRsub = Is.PtPs_Mach(Msub, gam)

res = {}
meshsim = mesh.unimesh(ncell=ncell, length=10.0)

def S(x):  # section law, throat is at x=5
    return 1 + (AsAc - 1.0) * (1.0 - np.exp(-0.5 * (x - 2.0) ** 2))

model = euler.nozzle(gamma=gam, sectionlaw=S)
nozz = sol.nozzle(model, S(meshsim.centers()), NPR=NPRsup)
finit = nozz.fdata(meshsim)
print(NPRsup, AsAc, Msup, Msub)

# solver / numerical model
bcL = {"type": "insub_cbc", "ptot": NPRsup, "rttot": 1.0}
bcR = {"type": bctype,   "p": 1.}
rhs = modeldisc.fvm(model, meshsim, muscl(vanleer), bcL=bcL, bcR=bcR)
solver = rk3ssp(meshsim, rhs)

#finit = rhs.fdata_fromprim([1.4, Msub, 1.])
cfl = 0.8
monitors = {"residual": {"frequency": 1}}
fsol = solver.solve(finit, cfl, stop={"maxit": nit_super}, monitors=monitors)

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (14,4))
monitors["residual"]["output"].semilogplot_it(ax=ax1)
fsol[-1].plot('mach', style='-', axes=ax2)
fsol[-1].plot('ptot', style='-', axes=ax3)
finit.plot('mach', style='--', axes=ax2)
finit.plot('ptot', style='--', axes=ax3)
plt.show()

# if verbose: solver.show_perf()
res["init_residual"] = monitors["residual"]["output"]._value[-1]
res["init_M9"] = fsol[-1].phydata("mach")[-1]

# RESTART
bcR = {"type": bctype, "p": 1.1*sw.Ps_ratio(Msup, gam)}
rhs = modeldisc.fvm(model, meshsim, muscl(vanleer), bcL=bcL, bcR=bcR)
solver = rk3ssp(meshsim, rhs)
try:
    fsol = solver.solve(fsol[-1], cfl, stop={"maxit": nit_tot}, monitors=monitors )
    solver.show_perf()
    res["simu_residual"] = monitors["residual"]["output"]._value[-1]
    res["simu_M9"] = fsol[-1].phydata("mach")[-1]
    # if verbose: print(res)
except:
    res["simu_residual"] = 1.0e9
    res["simu_M9"] = 0.0

print(res)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (14,4))
monitors["residual"]["output"].semilogplot_it(ax=ax1)
fsol[-1].plot('mach', style='-', axes=ax2)
fsol[-1].plot('ptot', style='-', axes=ax3)
plt.show()