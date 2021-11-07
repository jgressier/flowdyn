# -*- coding: utf-8 -*-
"""
xnum: package for spatial numerical methods
  extrapol1()
  extrapol2()
  extrapol3()
  extrapolk()
"""
__all__ = ['extrapol1', 'extrapol2', 'extrapol3', 'extrapolk', 'extrapol2d1',
        'muscl', 'minmod', 'vanalbada', 'vanleer', 'superbee']

import numpy as np
#import mesh

class virtualmeth():
    def __init__(self):
        self.gradmeth = 'none'
    
    # def interp_face(self, mesh, data, grad):
    #     pass
    
class extrapol1(virtualmeth):
    "first order method"
    def __init__(self):
        virtualmeth.__init__(self)

    def interp_face(self, mesh, data, grad='none'):
        "returns 2x (L/R) neq list of (ncell+1) nparray"
        nc = data[0].size
        if (mesh.ncell != nc): print(self.__class__+"/interp_face error: mismatch sizes")
        Ldata = []
        Rdata = []
        for i in range(len(data)):
            Ldata.append(np.zeros(nc+1))
            Rdata.append(np.zeros(nc+1))
            Ldata[i][1:]   = data[i][:]
            Rdata[i][0:-1] = data[i][:]
            #print 'L/R',Ldata[i].size, Rdata[i].size
        return Ldata, Rdata
        
class extrapol2(virtualmeth):
    "second order method without limitation, equivalent to extrapolk with k=-1"
    def __init__(self):
        virtualmeth.__init__(self)
        self.gradmeth = 'face'
        
    def interp_face(self, mesh, data, grad):
        "returns 2x (L/R) neq list of (ncell+1) nparray / except bound"
        nc = data[0].size
        if (mesh.ncell != nc): print(self.__class__+"/interp_face error: mismatch sizes")
        Ldata = []
        Rdata = []
        for i in range(len(data)):
            Ldata.append(np.zeros(nc+1))
            Rdata.append(np.zeros(nc+1))
            Ldata[i][1:]   = data[i][:] + grad[i][0:-1]*(mesh.xf[1:]  -mesh.xc[:])   # data are cell based index, grad are facebased index
            Rdata[i][0:-1] = data[i][:] + grad[i][1:]  *(mesh.xf[0:-1]-mesh.xc[:])
        return Ldata, Rdata

class extrapolk(virtualmeth):
    "second order method without limitation and k coefficient"
    def __init__(self, k):
        virtualmeth.__init__(self)
        self.gradmeth = 'face'
        self.kprec    = k
        
    def interp_face(self, mesh, data, grad):
        "returns 2x (L/R) neq list of (ncell+1) nparray / except bound"
        nc = data[0].size
        if (mesh.ncell != nc): print(self.__class__+"/interp_face error: mismatch sizes")
        Ldata = []
        Rdata = []
        for i in range(len(data)):
            Ldata.append(np.zeros(nc+1))
            Rdata.append(np.zeros(nc+1))
            # gradient are numbered at face
            Ldata[i][1:]   = data[i][:] + ((1-self.kprec)*grad[i][0:-1] +(1+self.kprec)*grad[i][1:])  /2*(mesh.xf[1:]  -mesh.xc[:])
            Rdata[i][0:-1] = data[i][:] + ((1-self.kprec)*grad[i][1:]   +(1+self.kprec)*grad[i][0:-1])/2*(mesh.xf[0:-1]-mesh.xc[:])
        return Ldata, Rdata

class extrapol2d1(virtualmeth):
    "first order method"
    def __init__(self):
        virtualmeth.__init__(self)
        self.gradmeth = 'face'

    def interp_face(self, mesh, data,  field, neq,xgrad='none',ygrad='none'):
        Ldata = []
        Rdata = []
        Ldata = field.zero_datalist(newdim=mesh.nbfaces())
        Rdata = field.zero_datalist(newdim=mesh.nbfaces())
        # reorder data for 1st order extrapolation, except for boundary conditions
        nx = mesh.nx
        ny = mesh.ny
        fshift = ny*(nx+1)
        for p in range(neq):
            if data[p].ndim == 2:
                # distribute cell states (by j rows) to i faces
                for j in range(ny):
                    Ldata[p][:,j*(nx+1)+1:(j+1)*(nx+1)] = data[p][:,j*nx:(j+1)*nx]
                    Rdata[p][:,j*(nx+1):(j+1)*(nx+1)-1] = data[p][:,j*nx:(j+1)*nx]
                # distribute cell states  (by j rows) to j faces (starting at index ny*(nx+1))
                for j in range(ny):
                    Ldata[p][:,fshift+(j+1)*nx:fshift+(j+2)*nx] = data[p][:,j*nx:(j+1)*nx]
                    Rdata[p][:,fshift+ j   *nx:fshift+(j+1)*nx] = data[p][:,j*nx:(j+1)*nx]
            else:
                # distribute cell states (by j rows) to i faces
                for j in range(ny):
                    Ldata[p][j*(nx+1)+1:(j+1)*(nx+1)] = data[p][j*nx:(j+1)*nx]
                    Rdata[p][j*(nx+1):(j+1)*(nx+1)-1] = data[p][j*nx:(j+1)*nx]
                # distribute cell states  (by j rows) to j faces (starting at index ny*(nx+1))
                for j in range(ny):
                    Ldata[p][fshift+(j+1)*nx:fshift+(j+2)*nx] = data[p][j*nx:(j+1)*nx]
                    Rdata[p][fshift+ j   *nx:fshift+(j+1)*nx] = data[p][j*nx:(j+1)*nx]
        return Ldata, Rdata

class extrapol2dk(virtualmeth):
    def __init__(self, k):
        virtualmeth.__init__(self)
        self.gradmeth = 'face'
        self.kprec    = k

    def interp_face(self, mesh, data,field, neq, xgrad, ygrad):
        Ldata = []
        Rdata = []
        Ldata = field.zero_datalist(newdim=mesh.nbfaces())
        Rdata = field.zero_datalist(newdim=mesh.nbfaces())
        kp = (1+self.kprec)/4.
        km = (1-self.kprec)/4.
        nx = mesh.nx
        ny = mesh.ny
        fshift = ny*(nx+1)
        for p in range(neq):
            if data[p].ndim == 2:
                for j in range(ny): # loop of cell rows
                    Ldata[p][:,j*(nx+1)+1:(j+1)*(nx+1)] = data[p][:,j*nx:(j+1)*nx] + km * xgrad[p][:,(nx+1)*j:j*(nx+1)+nx] + kp * xgrad[p][:,j*(nx+1)+1:(j+1)*(nx+1)]
                    Rdata[p][:,j*(nx+1):(j+1)*(nx+1)-1] = data[p][:,j*nx:(j+1)*nx] - km * xgrad[p][:,j*(nx+1)+1:(j+1)*(nx+1)] - kp * xgrad[p][:,(nx+1)*j:j*(nx+1)+nx]
                    
                for j in range(ny):  # loop of cell rows
                    Ldata[p][:,fshift+(j+1)*nx:fshift+(j+2)*nx] = data[p][:,j*nx:(j+1)*nx] + km * ygrad[p][:,j*nx:(j+1)*nx] + kp * ygrad[p][:,nx*(j+1):(j+2)*nx]
                    Rdata[p][:,fshift+ j   *nx:fshift+(j+1)*nx] = data[p][:,j*nx:(j+1)*nx] - km * ygrad[p][:,nx*(j+1):(j+2)*nx] - kp * ygrad[p][:,j*nx:(j+1)*nx]
                    
            else:
                # distribute cell states (by j rows) to i faces
                for j in range(ny): # loop of cell rows
                    Ldata[p][j*(nx+1)+1:(j+1)*(nx+1)] = data[p][j*nx:(j+1)*nx] + km * xgrad[p][(nx+1)*j:j*(nx+1)+nx] + kp * xgrad[p][j*(nx+1)+1:(j+1)*(nx+1)]
                    Rdata[p][j*(nx+1):(j+1)*(nx+1)-1] = data[p][j*nx:(j+1)*nx] - km * xgrad[p][j*(nx+1)+1:(j+1)*(nx+1)] - kp * xgrad[p][(nx+1)*j:j*(nx+1)+nx]
                    # distribute cell states  (by j rows) to j faces (starting at index ny*(nx+1))
                for j in range(ny): # loop of cell rows
                    Ldata[p][fshift+(j+1)*nx:fshift+(j+2)*nx] = data[p][j*nx:(j+1)*nx] + km * ygrad[p][j*nx:(j+1)*nx] + kp * ygrad[p][nx*(j+1):(j+2)*nx]
                    Rdata[p][fshift+ j   *nx:fshift+(j+1)*nx] = data[p][j*nx:(j+1)*nx] - km * ygrad[p][nx*(j+1):(j+2)*nx] - kp * ygrad[p][j*nx:(j+1)*nx]
        
        return Ldata, Rdata


class centered(extrapolk):
    "second order method without limitation, k=1 (Centered)"
    def __init__(self):
        extrapolk.__init__(self, k=1)
        
class fromm(extrapolk):
    "second order method without limitation, k=0 (Fromm)"
    def __init__(self):
        extrapolk.__init__(self, k=0.)
        
class quick(extrapolk):
    "second order method without limitation, k=1/2 (Quick)"
    def __init__(self):
        extrapolk.__init__(self, k=1./2.)
        
class extrapol3(extrapolk):
    "third order method without limitation, k=1/3"
    def __init__(self):
        extrapolk.__init__(self, k=1./3.)
        
# -----------------------------------------------------------------------------
# MUSCL van Leer schemes

def minmod(a,b):
    p = a*b
    return np.where(p <= 0., 0, np.where( a > 0, np.minimum(a, b), np.maximum(a,b) ) )

def vanalbada(a,b):
    p = a*b
    return np.where(p <= 1e-40, 0.,  p*(a+b)/(a**2+b**2+1e-20) )

def vanleer(a,b):
    p = a*b
    s = np.abs(a+b)+1.e-20
    return np.where(p <= 1e-40, 0., 2*p/s*np.sign(a) )
    #return np.where(p <= 0., 0, 2*p/(a+b) )

def superbee(a,b):
    p = a*b
    return np.where(p <= 0., 0, np.where( a > 0, np.minimum( 2*np.minimum(a, b), np.maximum(a, b)),
                                                np.maximum( 2*np.maximum(a, b), np.minimum(a, b)) ) )
    #alternate formula
    #return np.where(p <= 0, 0, np.where( a > 0, np.maximum( np.minimum(2*a, b), np.minimum(a, 2*b)),
    #                                            np.minimum( np.maximum(2*a, b), np.maximum(a, 2*b)) ) )

class muscl(virtualmeth):
    "second order MUSCL method"
    def __init__(self, limiter=minmod):
        virtualmeth.__init__(self)
        self.gradmeth = 'face'
        self.limiter  = limiter
#        print limiter
        
    def interp_face(self, mesh, data, grad):
        "returns 2x (L/R) neq list of (ncell+1) nparray / except bound"
        nc = data[0].size
        if (mesh.ncell != nc): print (self.__class__+"/interp_face error: mismatch sizes")
        Ldata = []
        Rdata = []
        for i in range(len(data)):
            Ldata.append(np.zeros(nc+1))
            Rdata.append(np.zeros(nc+1))
            # if limiter not symmetric, use centered gradient first
            Ldata[i][1:]   = data[i][:] + self.limiter( grad[i][1:],   grad[i][0:-1] ) *(mesh.xf[1:]  -mesh.xc[:])
            Rdata[i][0:-1] = data[i][:] + self.limiter( grad[i][0:-1], grad[i][1:]   ) *(mesh.xf[0:-1]-mesh.xc[:])
        return Ldata, Rdata

