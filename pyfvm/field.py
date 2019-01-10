# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:58:31 2013

@author: j.gressier
"""
import numpy as np
#import model
import mesh

class field():
    """
    define field: neq x nelem data
      model : number of equations
      nelem : number of cells (conservative and primitive data)
      qdata : list of neq nparray - conservative data 
      pdata : list of neq nparray - primitive    data
      bc    : type of boundary condition - "p"=periodic / "d"=Dirichlet 
    """
    def __init__(self, model, bc='p', nelem=100, bcvalues = []):
        self.model = model
        self.neq   = model.neq
        self.nelem = nelem
        self.qdata = []
        self.pdata = []
        self.time  = 0.
        self.bc    = bc        
        self.bcvalues = bcvalues        
        for i in range(self.neq+1):
            self.qdata.append(np.zeros(nelem))
            self.pdata.append(np.zeros(nelem))
            
    def cons2prim(self):
        self.pdata = self.model.cons2prim(self.qdata)
    
    def prim2cons(self):
        self.qdata = self.model.prim2cons(self.pdata) 
        
    def copy(self):
        new = field(self.model, self.bc, self.nelem, self.bcvalues)
        new.time  = self.time
        new.bc    = self.bc
        new.bcvalues = self.bcvalues
        new.nelem = self.nelem
        new.qdata = [ d.copy() for d in self.qdata ]
        new.pdata = [ d.copy() for d in self.pdata ]
        return new
        
class numfield(field):
    
    def __init__(self, f):
        self.model = f.model
        self.neq   = f.neq
        self.nelem = f.nelem
        self.bc    = f.bc
        self.bcvalues = f.bcvalues        
        self.qdata = [ d.copy() for d in f.qdata ]
        self.pdata = [ d.copy() for d in f.pdata ]
        self.time  = f.time

 
    def calc_grad(self, mesh):
        self.grad = []
        for d in self.pdata:
            g = np.zeros(mesh.ncell+1)
            g[1:-1] = (d[1:]-d[0:-1]) / (mesh.xc[1:]-mesh.xc[0:-1])
            self.grad.append(g)
    
    def interp_face(self, mesh, num):
        self.pL, self.pR = num.interp_face(mesh, self.pdata, self.grad)                
    
    def calc_bc(self):
        for i in range(self.neq):
            if self.bc == 'p':     #periodic boundary conditions
                for i in range(self.neq):
                    self.pL[i][0]          = self.pL[i][self.nelem] #=0
                    self.pR[i][self.nelem] = self.pR[i][0] #= 0
                    #print 'BC L/R',self.pL[i], self.pR[i]
            elif self.bc == 'd':   #dirichlet boundary conditions
                for i in range(self.neq):
                    self.pL[i][0]          = self.bcvalues[i][0] 
                    self.pR[i][self.nelem] = self.bcvalues[i][1]
            else:
                raise NameError("unknown BC condition: "+self.bc)
    
    def calc_bc_grad(self, mesh):
        for i in range(self.neq):
            self.grad[i][0] = self.grad[i][-1] = (self.pdata[i][0]-self.pdata[i][-1]) / (mesh.xc[0]+mesh.length-mesh.xc[-1])
            #print 'BC L/R',self.pL[i], self.pR[i]
    
    def calc_flux(self):
            self.flux = self.model.numflux(self.pL, self.pR)

    def calc_timestep(self, mesh, condition):
        if not self.pdata:
            #print self.qdata    #test
            return self.model.timestep(self.qdata, mesh.xf[1:self.nelem+1]-mesh.xf[0:self.nelem], condition)
        else:
            return self.model.timestep(self.pdata, mesh.xf[1:self.nelem+1]-mesh.xf[0:self.nelem], condition)
        
    def calc_res(self, mesh):
        self.residual = []
        for i in range(self.neq):
            self.residual.append(-(self.flux[i][1:self.nelem+1]-self.flux[i][0:self.nelem]) \
                                  /(mesh.xf[1:self.nelem+1]-mesh.xf[0:self.nelem]))
        return self.residual

    def add_res(self, time):
        for i in range(self.neq):
            #print i,self.qdata[i].size,time,self.residual[i].size
            self.qdata[i] += time*self.residual[i]  # time can be scalar or np.array

    def save_res(self):
        self.lastresidual = [ q.copy() for q in self.residual ]
                    
class scafield(field):
    def __init__(self, model, bc, nelem=100, bcvalues = []):
        field.__init__(self, model, bc, nelem=nelem, bcvalues=bcvalues)
            
    def scadata(self):
        return self.data[0]