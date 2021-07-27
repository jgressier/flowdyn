import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
#import matplotlib.patches as mpatch
import flowdyn.integration as tnum


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,6))
colors=colour_codes = map('C{}'.format, cycle(range(10)))
#
dt = 1.
x = np.r_[-3.5:.5:60j]/dt
y = np.r_[0:4.5:80j]/dt
X, Y = np.meshgrid(x, y)
vpgrid = dt*(X+Y*1j)
vpimag = dt*y*1j
#
for num in tnum.List_Explicit_Integrators:
    integrator = num(None, None)
    color=next(colors)
    ax1.contour(X,Y,abs(integrator.propagator(vpgrid)), levels=[1], linewidths=2, colors=color) #, colors='darkorange') # contour() accepts complex values
    prop_num = integrator.propagator(vpimag)
    prop_th = np.exp(vpimag)
    error   = np.log(prop_num/prop_th)
    ax2.plot(y, np.real(error), linestyle='solid', color=color, label=integrator.__class__.__name__)
    ax3.plot(y, np.imag(error), linestyle='dashed', color=color,)
#plt.contourf(X,Y,abs(integrator.propagator(vp)), ) # contour() accepts complex values
ax1.axis('equal')
ax1.set_ylim(0, np.max(y))
ax2.legend() #bbox=dict(boxstyle='round4', fc="w", ec="k"))
plt.show()