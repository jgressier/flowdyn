# -*- coding: utf-8 -*-
"""
xnum: package for spatial numerical methods
"""

import numpy as np
import mesh

class virtualmeth():
    def __init__(self):
        self.gradmeth = 'none'
    
    def interp_face(self, mesh, data):
        pass
    
class extrapol1(virtualmeth):
    "first order method"
    def interp_face(self, mesh, data, grad='none'):
        "returns 2x (L/R) neq list of (ncell+1) nparray"
        nc = data[0].size
        if (mesh.ncell <> nc): print self.__class__+"/interp_face error: mismatch sizes"
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
    "second order method without limitation"
    def __init__(self):
        self.gradmeth = 'face'
        
    def interp_face(self, mesh, data, grad):
        "returns 2x (L/R) neq list of (ncell+1) nparray / except bound"
        nc = data[0].size
        if (mesh.ncell <> nc): print self.__class__+"/interp_face error: mismatch sizes"
        Ldata = []
        Rdata = []
        for i in range(len(data)):
            Ldata.append(np.zeros(nc+1))
            Rdata.append(np.zeros(nc+1))
            Ldata[i][1:]   = data[i][:] + grad[i][0:-1]*(mesh.xf[1:]  -mesh.xc[:])
            Rdata[i][0:-1] = data[i][:] + grad[i][1:]  *(mesh.xf[0:-1]-mesh.xc[:])
        return Ldata, Rdata

class extrapolk(virtualmeth):
    "second order method without limitation and k coefficient"
    def __init__(self, k):
        self.gradmeth = 'face'
        self.kprec    = k
        
    def interp_face(self, mesh, data, grad):
        "returns 2x (L/R) neq list of (ncell+1) nparray / except bound"
        nc = data[0].size
        if (mesh.ncell <> nc): print self.__class__+"/interp_face error: mismatch sizes"
        Ldata = []
        Rdata = []
        for i in range(len(data)):
            Ldata.append(np.zeros(nc+1))
            Rdata.append(np.zeros(nc+1))
            Ldata[i][1:]   = data[i][:] + ((1-self.kprec)*grad[i][0:-1] +(1+self.kprec)*grad[i][1:])  /2*(mesh.xf[1:]  -mesh.xc[:])
            Rdata[i][0:-1] = data[i][:] + ((1-self.kprec)*grad[i][1:]   +(1+self.kprec)*grad[i][0:-1])/2*(mesh.xf[0:-1]-mesh.xc[:])
        return Ldata, Rdata

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
        
