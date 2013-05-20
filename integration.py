# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:39:03 2013

@author: j.gressier
"""

import numpy as np
from field import *

class timemodel():
    def __init__(self, mesh, num):
        self.mesh  = mesh
        self.num   = num
        
    def calcrhs(self, field):
        field.cons2prim()
        field.calc_grad(self.mesh)
        field.calc_bc_grad(self.mesh)
        field.interp_face(self.mesh, self.num)
        field.calc_bc()
        field.calc_flux()
        return field.calc_res(self.mesh)
 
    def step():
        print "not implemented for virtual class"

    def solve(self, field, condition, tsave):
        self.nit       = 0
        self.condition = condition
        itfield = numfield(field)
        results = []
        for t in np.arange(tsave.size):
            endcycle = 0
            while endcycle == 0:
                dtloc = itfield.calc_timestep(self.mesh, condition)
                dtloc = min(dtloc)
                if itfield.time+dtloc >= tsave[t]:
                    endcycle = 1
                    dtloc    = tsave[t]-itfield.time
                self.nit += 1
                itfield.time += dtloc
                itfield = self.step(itfield, dtloc)
            itfield.cons2prim()
            results.append(itfield.copy())
        return results

    
class time_explicit(timemodel):
    def step(self, field, dtloc):
        self.calcrhs(field)
        field.add_res(dtloc)
        return field
    
class time_rk2(timemodel):
    def step(self, field, dtloc):
        reffield = numfield(field)
        self.calcrhs(field)
        field.add_res(dtloc/2)
        self.calcrhs(field)
        reffield.residual = field.residual
        reffield.add_res(dtloc)
        return reffield