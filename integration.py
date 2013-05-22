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
    
class rkmodel(timemodel):
    def step(self, field, dtloc, butcher):
        #butcher = [ np.array([1.]), \
        #            np.array([0.25, 0.25]), \
        #            np.array([1., 1., 4.])/6. ]
        prhs = []
        pfield = numfield(field)            
        for pcoef in butcher:
            # compute residual of previous stage and memorize it in prhs[]
            prhs.append([ q.copy() for q in self.calcrhs(pfield)])
            # revert to initial step
            pfield.qdata = [ q.copy() for q in field.qdata ]
            # aggregate residuals
            for qf in pfield.residual:
                qf *= pcoef[-1]
            for i in range(pcoef.size-1):
                for q in range(pfield.neq):
                    pfield.residual[q] += pcoef[i]*prhs[i][q]
            pfield.add_res(dtloc)        
        return pfield

class time_rk2(timemodel):
    def step(self, field, dtloc):
        reffield = numfield(field)
        self.calcrhs(field)
        field.add_res(dtloc/2)
        self.calcrhs(field)
        reffield.residual = field.residual
        reffield.add_res(dtloc)
        return reffield

class time_rk3ssp(rkmodel):
    def step(self, field, dtloc):
        butcher = [ np.array([1.]), \
                    np.array([0.25, 0.25]), \
                    np.array([1., 1., 4.])/6. ]
        return rkmodel.step(self, field, dtloc, butcher)

class time_rk4(rkmodel):
    def step(self, field, dtloc):
        butcher = [ np.array([0.5]), \
                    np.array([0., 0.5]), \
                    np.array([0., 0., 1.]), \
                    np.array([1., 2., 2., 1.])/6. ]
        return rkmodel.step(self, field, dtloc, butcher)


