# -*- coding: utf-8 -*-
"""
time integration methods (class)
available are
explicit or forwardeuler
rk2
rk3ssp
rk4
implicit or ba
trapezoidal or cranknichlson
"""
import math
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
        itfield.cons2prim()
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
                if dtloc > np.spacing(dtloc):
                    itfield = self.step(itfield, dtloc)
            itfield.cons2prim()
            results.append(itfield.copy())
        return results

    
class explicit(timemodel):
    def step(self, field, dtloc):
        self.calcrhs(field)
        field.add_res(dtloc)
        return field

class forwardeuler(explicit):
    pass
    
#--------------------------------------------------------------------
# RUNGE KUTTA MODELS
#--------------------------------------------------------------------
    
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

class rk2(timemodel):
    def step(self, field, dtloc):
        reffield = numfield(field)
        self.calcrhs(field)
        field.add_res(dtloc/2)
        self.calcrhs(field)
        reffield.residual = field.residual
        reffield.add_res(dtloc)
        return reffield

class rk3ssp(rkmodel):
    def step(self, field, dtloc):
        butcher = [ np.array([1.]), \
                    np.array([0.25, 0.25]), \
                    np.array([1., 1., 4.])/6. ]
        return rkmodel.step(self, field, dtloc, butcher)

class rk4(rkmodel):
    def step(self, field, dtloc):
        butcher = [ np.array([0.5]), \
                    np.array([0., 0.5]), \
                    np.array([0., 0., 1.]), \
                    np.array([1., 2., 2., 1.])/6. ]
        return rkmodel.step(self, field, dtloc, butcher)

#--------------------------------------------------------------------
# IMPLICIT MODELS
#--------------------------------------------------------------------

class implicitmodel(timemodel):
    def step(self, field, dtloc):
        print "not implemented for virtual implicit class"
        
    def calc_jacobian(self, field):
        if ((field.model.islinear == 1) and (hasattr(self, "jacobian_use"))):
            return
        self.neq = field.neq
        self.dim = self.neq * field.nelem
        self.jacobian = np.zeros([self.dim, self.dim])
        #self.jact     = np.zeros([self.dim, self.dim])
        eps = [ math.sqrt(np.spacing(1.))*np.sum(np.abs(q))/field.nelem for q in field.qdata ] 
        refrhs = [ qf.copy() for qf in self.calcrhs(field) ]
        #print 'refrhs',refrhs
        for i in range(field.nelem):    # for all variables (nelem*neq)
            for q in range(self.neq):
                dfield = numfield(field)
                dfield.qdata[q][i] += eps[q]
                drhs = [ qf.copy() for qf in self.calcrhs(dfield) ]
                for qq in range(self.neq):
                    #self.jacobian[i*self.neq+q,qq::self.neq] = (drhs[qq]-refrhs[qq])/eps[q] # working
                    self.jacobian[qq::self.neq,i*self.neq+q] = (drhs[qq]-refrhs[qq])/eps[q]
        self.jacobian_use = 0
        #print self.jacobian - self.jact.transpose()
        return self.jacobian

    def solve_implicit(self, field, dtloc, invert=np.linalg.solve, theta=1., xi=0):
        ""
        diag = np.repeat(np.ones(field.nelem)/dtloc, self.neq)   # dtloc can be scalar or np.array
        mat = (1+xi)*np.diag(diag)-theta*self.jacobian#.transpose()
        rhs = np.concatenate(field.residual)
        if xi != 0: 
            rhs += xi* np.concatenate(field.lastresidual)
        newrhs = np.linalg.solve(mat, rhs)
        field.residual = [ newrhs[iq::self.neq]/dtloc for iq in range(self.neq) ]
    
class implicit(implicitmodel):
    def step(self, field, dtloc):                
        self.calc_jacobian(field)
        self.calcrhs(field)
        self.solve_implicit(field, dtloc)
        field.add_res(dtloc)
        return field
    
class backwardeuler(implicit):
    pass

class trapezoidal(implicitmodel):
    def step(self, field, dtloc):                
        self.calc_jacobian(field)
        self.calcrhs(field)
        self.solve_implicit(field, dtloc, theta=.5)
        field.add_res(dtloc)
        return field

class cranknicolson(trapezoidal):
    pass

class gear(trapezoidal):
    def step(self, field, dtloc):
        if not hasattr(field, 'lastresidual'):      # if starting integration (missing last residual)
            field = trapezoidal.step(self, field, dtloc)
            field.save_res()
            return field
        self.calc_jacobian(field)
        self.calcrhs(field)
        self.solve_implicit(field, dtloc, theta=1., xi=.5)
        field.add_res(dtloc)
        field.save_res()
        return field

#--------------------------------------------------------------------
# LOW STORAGE MODELS; rk1 / rk22 / rk3lsw / rk3ssp / rk4
#--------------------------------------------------------------------       

class LowStorageRKmodel(timemodel):

    def __init__(self, mesh, num):
        self.mesh = mesh
        self.num  = num

    def solve(self, field, condition, tsave):
        self.nit       = 0
        self.condition = condition
        self.neq = field.neq #to have the number of equations available
        self.nelem = field.nelem #to have the number of elements available
        self.new_rhs=np.zeros((self.nstage,self.neq,self.nelem))       

        itfield = numfield(field)
        itfield.cons2prim()
        results = []
        for t in np.arange(tsave.size):
            endcycle = 0
            while endcycle == 0:
                dtloc  = itfield.calc_timestep(self.mesh, condition)
                dtglob = min(dtloc)
                self.nit += 1
                itfield.nit = self.nit
                itfield.time += dtglob
                if itfield.time+dtglob >= tsave[t]:
                    endcycle = 1
                    dtglob    = tsave[t]-itfield.time
                if dtglob > np.spacing(dtglob):
                    self.new_rhs[:,:,:]=0.0
                    for irkstep in range(self.nstage):
                        itfield = self.step(itfield,dtglob,irkstep) 
                        itfield.cons2prim()
            results.append(itfield.copy())
        return results

    def step(self, field, dt,irkstep):
        self.calcrhs(field)
        for j in range(self.neq):
            for k in range(self.nelem):
                self.new_rhs[irkstep,j,k] = field.residual[j][k]           
        self.add_res(field,dt, irkstep)
        return field

    def add_res(self,field, dt, irkstep):      
        for rk_coeff_index in range(irkstep+1):
            for i in range(self.neq):
                field.qdata[i] += dt * self.RKcoeff[irkstep, rk_coeff_index] * self.new_rhs[rk_coeff_index,i,:]  # time can be scalar or np.array           

class LSrk1(LowStorageRKmodel):
    def __init__(self, mesh, num):

        self.mesh    = mesh
        self.num     = num
        self.nstage  = 1
        self.RKcoeff = np.array ([[1.]])

class LSrk22(LowStorageRKmodel):
    def __init__(self, mesh, num):

        self.mesh    = mesh
        self.num     = num
        self.nstage  = 2
        self.RKcoeff = np.array ([[0.5, 0.],
                                  [-0.5, 1.]])

class LSrk3ssp(LowStorageRKmodel):
    def __init__(self, mesh, num):

        self.mesh    = mesh
        self.num     = num
        self.nstage  = 3
        self.RKcoeff = np.array([[1., 0., 0.],
                                 [-3./4., 1./4., 0.],
                                 [-1./12., -1./12., 2./3.]])

class LSrk3lsw(LowStorageRKmodel):
    def __init__(self, mesh, num):

        self.mesh    = mesh
        self.num     = num
        self.nstage  = 3
        self.RKcoeff = np.array ([[8./15., 0., 0.],
                                  [-17./60., 5./12., 0.],
                                  [0., -5./12., 3./4.]])

class LSrk4(LowStorageRKmodel):
    def __init__(self, mesh, num):

        self.mesh    = mesh
        self.num     = num
        self.nstage  = 4
        self.RKcoeff = np.array([[1./2., 0., 0., 0.],
                                 [-1./2., 1./2., 0., 0.],
                                 [0., -1./2., 1., 0.],
                                 [1./6., 1./3., -2./3., 1./6.]])                                                       