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
        field.interp_face(self.mesh, self.num)
        field.calc_bc()
        field.calc_flux()
        field.calc_res(self.mesh)
 
    def step():
        print "not implemented for virtual class"

    def solve(self, field, condition, tsave):
        itfield = numfield(field)
        results = []
        for t in np.arange(tsave.size):
            endcycle = 0
            while endcycle == 0:
                dtloc = itfield.calc_timestep(self.mesh, condition)
                dtloc = min(dtloc)
                if itfield.time+dtloc > tsave[t]:
                    endcycle = 1
                    dtloc    = tsave[t]-itfield.time
                itfield.time += dtloc
                self.step(itfield, dtloc)
            itfield.cons2prim()
            results.append(itfield.copy())
        return results

    
class timeexplicit(timemodel):
    def step(self, field, dtloc):
        self.calcrhs(field)
        field.add_res(dtloc)