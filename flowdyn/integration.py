# -*- coding: utf-8 -*-
"""
time integration methods (class)
available are
explicit or forwardeuler
rk2
rk3ssp
rk4
implicit or backwardeuler
trapezoidal or cranknicolson
"""
import math
import numpy as np
import sys
import time
import flowdyn.field as field

#--------------------------------------------------------------------
class fakemodel():
    def __init__(self):
        self.neq = 1
        self.shape = [1]
class fakemesh():
    def __init__(self):
        self.ncell = 1
class fakedisc():
    def __init__(self, z):
        self.z = z
    def rhs(self, f):
        return [f.data[0]*self.z]

#--------------------------------------------------------------------
# generic model
#--------------------------------------------------------------------

class timemodel():
    def __init__(self, mesh, modeldisc):
        self.mesh      = mesh
        self.modeldisc = modeldisc
        self._cputime  = 0.
        self._nit      = 0
        
    def calcrhs(self, field):
        self.residual = self.modeldisc.rhs(field)
 
    def step(self, f, dt):
        raise NameError("not implemented for virtual class")

    def add_res(self, f, dt, subtimecoef = 1.0):
        f.time += np.min(dt)*subtimecoef
        for i in range(f.neq):
            #print i,self.qdata[i].size,time,self.residual[i].size
            f.data[i] += dt*self.residual[i]  # time can be scalar or np.array

    def save_res(self):
        self.lastresidual = [ q.copy() for q in self.residual ]

    def solve(self, f, condition, tsave, flush=None):
        if float(sys.version[:3]) >= 3.3: # or 3.8
            myclock = time.process_time
        else:
            myclock = time.clock
        start = myclock()
        self._nit      = 0
        self.condition = condition
        itfield = f.copy()
        if flush:
            alldata = [ d for d in itfield.data ]
        results = []
        for t in np.arange(len(tsave)):
            endcycle = 0
            while endcycle == 0:
                dtloc = self.modeldisc.calc_timestep(itfield, condition)
                dtloc = min(dtloc)
                if itfield.time+dtloc >= tsave[t]:
                    endcycle = 1
                    dtloc    = tsave[t]-itfield.time
                self._nit += 1
                if dtloc > np.spacing(dtloc):
                    self.step(itfield, dtloc)
                if flush:
                    for i, q in zip(range(len(alldata)), itfield.data):
                        alldata[i] = np.vstack((alldata[i], q))
            results.append(itfield.copy())
        self._cputime = myclock()-start
        if flush:
            np.save(flush, alldata)
        return results

    def nit(self):
        return self._nit

    def cputime(self):
        return self._cputime

    def show_perf(self):
        print("cpu time computation ({0:d} it) : {1:.3f}s\n  {2:.2f} µs/cell/it".format(
            self._nit, self._cputime, self._cputime*1.e6/self._nit/self.modeldisc.nelem))

    def propagator(self, z):
        # save actual modeldisc
        saved_model = self.modeldisc
        self.modeldisc = fakedisc(z)
        # make virtual field
        f = field.fdata(fakemodel(), fakemesh(), [0*z+1.])
        self.step(f, dtloc=1.)
        # get back actual modeldisc
        self.modeldisc = saved_model
        return f.data[0]
        
#--------------------------------------------------------------------

class explicit(timemodel):
    def step(self, field, dtloc):
        self.calcrhs(field)
        self.add_res(field, dtloc)
        return 

class forwardeuler(explicit): # alias of explicit
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
        pfield = field.copy()
        for pcoef in butcher:
            subtimecoef = np.sum(pcoef)
            # compute residual of previous stage and memorize it in prhs[]
            self.calcrhs(pfield) # result in self.residual
            prhs.append([ q.copy() for q in self.residual])
            # revert to initial step
            #pfield.data = [ q.copy() for q in field.data ]
            pfield = field.copy()
            # aggregate residuals
            for qf in self.residual:
                qf *= pcoef[-1]
            for i in range(pcoef.size-1):
                for q in range(pfield.neq):
                    self.residual[q] += pcoef[i]*prhs[i][q]
            # substep
            self.add_res(pfield, dtloc, subtimecoef)
        field.set(pfield)
        return 

class rk2(timemodel):
    def step(self, field, dtloc):
        pfield = field.copy()
        self.calcrhs(pfield)
        self.add_res(pfield, dtloc/2)
        self.calcrhs(pfield)
        #self.residual = field.residual
        self.add_res(field, dtloc)
        return 

class rk3ssp(rkmodel):
    _butcher = [ np.array([1.]), \
                 np.array([0.25, 0.25]), \
                 np.array([1., 1., 4.])/6. ]
    def step(self, field, dtloc):
        return rkmodel.step(self, field, dtloc, self._butcher)

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
        print("not implemented for virtual implicit class")
        
    def calc_jacobian(self, field, epsdiff=1.e-6):
        """
            jacobian matrix dR/dQ of dQ/dt=R(Q) is computed as successive columns by finite difference of R(Q+dQ)
            ordering is ncell x neq (neq is the fast index)
        """
        if ((field.model.islinear == 1) and (hasattr(self, "jacobian_use"))):
            return
        self.neq = field.neq
        self.dim = self.neq * field.nelem
        self.jacobian = np.zeros([self.dim, self.dim])
        eps = [ epsdiff*math.sqrt(np.spacing(1.))*np.sum(np.abs(q))/field.nelem for q in field.data ] 
        self.calcrhs(field)
        refrhs = [ qf.copy() for qf in self.residual ]
        for i in range(field.nelem):    # for all variables (nelem*neq)
            for q in range(self.neq):
                dfield = field.copy()
                dfield.data[q][i] += eps[q]
                self.calcrhs(dfield)
                drhs = [ qf.copy() for qf in  self.residual ]
                for qq in range(self.neq):
                    self.jacobian[qq::self.neq,i*self.neq+q] = (drhs[qq]-refrhs[qq])/eps[q]
        self.jacobian_use = 0
        return self.jacobian

    def solve_implicit(self, field, dtloc, invert=np.linalg.solve, theta=1., xi=0):
        ""
        diag = np.repeat(np.ones(field.nelem)/dtloc, self.neq)   # dtloc can be scalar or np.array, neq is the fast index
        mat = (1+xi)*np.diag(diag)-theta*self.jacobian
        rhs = np.zeros((self.dim))
        for q in range(self.neq):
            rhs[q::self.neq] = self.residual[q]
        if xi != 0: 
            for q in range(self.neq):
                rhs[q::self.neq] += xi* self._lastresidual[q]
        newrhs = invert(mat, rhs)
        self.residual = [ newrhs[iq::self.neq]/dtloc for iq in range(self.neq) ]
    
class implicit(implicitmodel):
    """
        make an Euler implicit or backward Euler step: Qn+1 - Qn = Rn+1 
    """
    def step(self, field, dtloc):           
        self.calc_jacobian(field)
        self.calcrhs(field)                  # compute and define self.residual
        self.solve_implicit(field, dtloc)    # save self.residual
        self.add_res(field, dtloc)
        return
    
class backwardeuler(implicit):
    pass

class trapezoidal(implicitmodel):
    """
        make an 2nd order (centered) Crank-Nicolson step: Qn+1 - Qn = .5*(Rn + Rn+1)
    """
    def step(self, field, dtloc):                
        self.calc_jacobian(field)
        self.calcrhs(field)
        self.solve_implicit(field, dtloc, theta=.5)
        self.add_res(field, dtloc)
        return

class cranknicolson(trapezoidal):
    pass

class gear(trapezoidal):
    """
        make an 2nd order backward step (Gear): (3Qn+1 - 4Qn + Qn-1)/3 = 2/3* Rn+1
        Using Rn+1 = Rn + A*(Qn+1-Qn), linearized form is (3I-2A)(Qn+1-Qn)=2Rn+(Qn-Qn-1)
    """
    def step(self, field, dtloc):
        if not hasattr(self, '_lastresidual'):      # if starting integration (missing last residual), so use 2nd order trapezoidal/cranknicolson
            trapezoidal.step(self, field, dtloc)
            self.add_res(field, dtloc)
        else:
            self.calc_jacobian(field)
            self.calcrhs(field)
            self.solve_implicit(field, dtloc, theta=1., xi=.5)
            self.add_res(field, dtloc)
        self._lastresidual = self.residual
        return 

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
                if itfield.time+dtglob >= tsave[t]:
                    endcycle = 1
                    dtglob    = tsave[t]-itfield.time
                if dtglob > np.spacing(dtglob):
                    self.new_rhs[:,:,:]=0.0
                    for irkstep in range(self.nstage):
                        itfield = self.step(itfield,dtglob,irkstep) 
                        itfield.cons2prim()
                itfield.time += dtglob
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
