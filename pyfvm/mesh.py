# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:42:29 2013

@author: j.gressier
"""
import numpy as np

class virtualmesh():
    " virtual class for a domain and its mesh"
    def __init__(self, ncell=0, length=0.):
        self.ncell  = ncell
        self.length = length

    def centers(self):
        "compute centers of cells in a mesh"
        xc = np.zeros(self.ncell)
        for i in np.arange(self.ncell):
            xc[i] = (self.xf[i]+self.xf[i+1])/2.
        return xc

    def dx(self):
        "compute cell sizes in a mesh"
        dx = np.zeros(self.ncell)
        for i in np.arange(self.ncell):
            dx[i] = (self.xf[i+1]-self.xf[i])
        return dx

    def __repr__(self):
        print "length : ", self.length
        print "ncell  : ", self.ncell
        dx = self.dx()
        print "min dx : ", dx.min()
        print "max dx : ", dx.max()


class unimesh(virtualmesh):
    " class defining a uniform mesh: ncell and length"
    def __init__(self, ncell=100, length=1.):
        virtualmesh.__init__(self, ncell, length)
        self.xf     = np.linspace(0., length, ncell+1)
        self.xc     = self.centers()

class refinedmesh(virtualmesh):
    " class defining a domain and its mesh: ncell and length"
    def __init__(self, ncell=100, length=1., ratio=2.):
        virtualmesh.__init__(self, ncell, length)
        dx1 = 2* length / ((1.+ratio)*ncell)
        dx2 = ratio*dx1
        nc1 = ncell/2
        nc2 = ncell-nc1
        self.xf = np.append(
                    np.linspace(0., dx1*nc1, nc1+1)[0:-1],
                    np.linspace(dx1*nc1, length, nc2+1) )
        self.xc = self.centers()

