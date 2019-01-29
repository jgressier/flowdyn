# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:58:31 2013

@author: j.gressier
"""
import numpy as np
import matplotlib.pyplot as plt
#import model
#import mesh

class fdata():
    """
    define field: neq x nelem data
      model : number of equations
      nelem : number of cells (conservative and primitive data)
      qdata : list of neq nparray - conservative data 
      pdata : list of neq nparray - primitive    data
      bc    : type of boundary condition - "p"=periodic / "d"=Dirichlet 
    """
    def __init__(self, model, mesh, data=None, t=0.):
        self.model = model
        self.neq   = model.neq
        self.mesh  = mesh
        self.nelem = mesh.ncell
        self.time  = t
        if data!=None:
            self.data = [ np.ones(self.nelem)*d for d in data ]
        else:
            self.data = [ np.zeros(self.nelem) ] * self.neq
            # for i in range(self.neq):
            #     self.data.append(np.zeros(nelem))
                    
    def copy(self):
        new = fdata(self.model, self.mesh, self.data)
        new.time  = self.time
        # new.mesh  = self.mesh
        # new.nelem = self.nelem
        # new.data = [ d.copy() for d in self.data ]
        return new

    def set(self, f):
        self.__init__(f.model, f.mesh, f.data)
        self.time = f.time

    def phydata(self, name):
        return self.model.nameddata(name, self.data)

    def plot(self, name, style='o'):
        return plt.plot(self.mesh.centers(), self.phydata(name), style)

    def set_plotdata(self, line, name):
        line.set_data(self.mesh.centers(), self.phydata(name))
        return
