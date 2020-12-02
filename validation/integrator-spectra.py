import numpy as np
import matplotlib.pyplot as plt
import pyfvm.integration as tnum

integrator = tnum.rk3ssp(None, None)

dt = 1.
x = np.r_[-2.8:.25:30j]/dt
y = np.r_[-2.8:2.8:60j]/dt
X, Y = np.meshgrid(x, y)
vp = dt*(X+Y*1j)
plt.contour(X,Y,abs(integrator.propagator(vp)), levels=[1], linewidths=3, colors='darkorange') # contour() accepts complex values
#plt.contourf(X,Y,abs(integrator.propagator(vp)), ) # contour() accepts complex values
plt.axis('equal')
plt.show()