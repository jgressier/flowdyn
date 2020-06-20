# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:42:29 2013

@author: j.gressier
"""

import sys
import math
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
        print("length : ", self.length)
        print("ncell  : ", self.ncell)
        dx = self.dx()
        print("min dx : ", dx.min())
        print("max dx : ", dx.max())

class nonunimesh(virtualmesh):
    " class defining a domain and a non-uniform mesh"
    #non-uniform symmetric mesh
        
    def __init__(self, length, nclass, ncell0, periods):

        self.periods = periods        
        self.length = length*periods
        self.nclass = nclass
        self.ncell0 = ncell0
                
        regions = 2*nclass
        rlength = length/regions
        
        rcells = np.zeros(nclass)
        for r in range(nclass):
            rcells[r]=ncell0*2**r
      
        for i in range(periods):
            if i == 0:
                self.xf = np.linspace(0.,rlength, rcells[nclass-1]+1) #creates linspace array of rcell cells
            else:
                idel = len(self.xf)
                self.xf = np.hstack((self.xf,np.linspace(i*length,rlength+i*length,rcells[nclass-1]+1))) #stacks another linspace array
                self.xf = np.delete(self.xf,idel) #deleting item in index ncell1 due to duplication  

            for r in range(nclass-2,-1,-1):  #creates from left to right till the middle
                idel = len(self.xf)
                self.xf = np.hstack((self.xf,np.linspace((nclass-r-1)*rlength+i*length,(nclass-r)*rlength+i*length,rcells[r]+1))) #stacks another linspace array
                self.xf = np.delete(self.xf,idel) #deleting item in index ncell1 due to duplication   
            
            for r in range(nclass):        #creates from left to right from the middle
                idel = len(self.xf)
                self.xf = np.hstack((self.xf,np.linspace((nclass+r)*rlength+i*length,(nclass+r+1)*rlength+i*length,rcells[r]+1))) #stacks another linspace array
                self.xf = np.delete(self.xf,idel) #deleting item in index ncell1 due to duplication               
        
        self.ncell = len(self.xf)-1
        self.xc = self.centers()
        
