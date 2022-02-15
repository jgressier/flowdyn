import numpy as np
import flowdyn.mesh  as mesh
import flowdyn.modelphy.convection as conv
import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
import flowdyn.xnum  as xnum
import flowdyn.integration as tnum


curmesh = mesh.unimesh(ncell=50, length=1.)
convmodel = conv.model(convcoef=1.)

def init_sinperk(mesh, k):
    return np.sin(2*k*np.pi/mesh.length*mesh.centers())

xsch = xnum.extrapol3()

tottime = 10.
breaktime = 5.
cfl     = .5
finit = field.fdata(convmodel, curmesh, [ init_sinperk(curmesh, k=4) ] )
rhs = modeldisc.fvm(convmodel, curmesh, xsch)
solver = tnum.rk4(curmesh, rhs)
nsol = 10+1 # every second
tsave = np.linspace(0, tottime, nsol, endpoint=True)
#
stop_directive = { 'tottime': breaktime }
fsol0 = solver.solve(finit, cfl, tsave, stop=stop_directive)
assert len(fsol0) < nsol # end before expected by tsave
assert not fsol0[-1].isnan()
assert fsol0[-1].time == breaktime
#
stop_directive = { 'tottime': tottime }
fsol = solver.restart(fsol0[-1], cfl, tsave, stop=stop_directive, directives={'verbose':1})
print(len(fsol),nsol) # only last snapshots
assert not fsol[-1].isnan()
assert fsol[-1].time == tottime