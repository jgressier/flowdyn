# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:42:29 2013

@author: j.gressier
"""
import numpy as np

class unimesh():
    " class defining a domain and its mesh: ncell and length"
    def __init__(self, ncell=100, length=1.):
        self.ncell  = ncell
        self.length = length
        self.xf     = np.linspace(0., length, ncell+1)
        self.xc     = self.centers()

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