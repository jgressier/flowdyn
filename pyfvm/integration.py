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

class timemodel():
    def __init__(self, mesh, modeldisc):
        self.mesh      = mesh
        self.modeldisc = modeldisc
        self._cputime  = 0.
        self._nit      = 0
        
    def calcrhs(self, field):
        self.residual = self.modeldisc.rhs(field)
 
    def step():
        print("not implemented for virtual class")

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
    
class explicit(timemodel):
    def step(self, field, dtloc):
        self.calcrhs(field)
        self.add_res(field, dtloc)
        return 

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

#--------------------------------------------------------------------
# ASYNC LOW STORAGE MODELS; rk1 / rk22 / rk3lsw / rk3ssp / rk4
#--------------------------------------------------------------------       

class AsyncLowStorageRKmodel(timemodel):

    def __init__(self, mesh, num):
        self.mesh    = mesh
        self.num     = num

    def solve(self, field, condition, tsave):
        self.nit        = 0
        self.condition  = condition
        self.bc         = field.bc
        self.asyncsq    = 0
        self.neq        = field.neq #to have the number of equations available
        self.nelem      = field.nelem #to have the number of elements available
        self.new_rhs    = np.zeros((self.nstage,self.neq,self.nelem))       
        self.async_flux = np.zeros((self.neq,self.nelem+1))

        global adjcells, incells, interface, classes, cell_class, minclass, maxclass

        itfield = numfield(field)
        itfield.cons2prim()
        results = []
        # flag = False
        for t in np.arange(tsave.size):
            endcycle = 0
            while endcycle == 0:
                dtloc  = itfield.calc_timestep(self.mesh, condition)
                # Computation of async arrays
                [nc, DT, classes, cell_class, adjcells,interface, self.async_seq] = self.cell_classify(dtloc)
                self.cell_class = cell_class
                minclass = min(cell_class) #coarsest class
                maxclass = max(cell_class) #finest class
                dtglob = DT[minclass] #global synchronization time step #if set to min(dtloc), should run as classic solve() 

                # if flag is True:          # test 
                #     print "time steps:    ", np.array_str(DT)              #test 
                #     print "cell classes:  ", classes                       #test
                #     print "cells' class:  ", np.array_str(cell_class)      #test
                #     print "interface:     ", interface
                #     raw_input('Press <ENTER> to continue')                 #test
                #     flag = False  

                self.nit += 1
                itfield.nit = self.nit
                if itfield.time+dtglob >= tsave[t]:
                    endcycle = 1
                    DT[minclass] =  tsave[t]-itfield.time     #if time is greater than total time
                    dtglob = DT[minclass]                     #global timestep set to C0 timestep    
                    for k in range(minclass,nc):              #set every class k timestep as well
                        DT[k+1] = 0.5 * DT[k]                 #as the half of the previous
                if dtglob > np.spacing(dtglob):
                    self.new_rhs[:,:,:]  = 0.0
                    self.async_flux[:,:] = 0.0
                    #
                    # Local asynchronous timesteps
                    #
                    for iasync in range(len(self.async_seq)):
                        iclass = self.async_seq[iasync] 
                        idtloc = DT[iclass]
                        for irkstep in range(self.nstage):
                            itfield = self.step(itfield, idtloc, irkstep, iclass, iasync) 
                            itfield.cons2prim()
                itfield.time += dtglob
            results.append(itfield.copy())
        return results

    def step(self, field, dtloc, irkstep, iclass, iasync):

        self.calcrhs(field)

        #=== CORRECTTION FOR THE FIRST ASYNC ALGO :: self.asyncsq = 0 :=> [2 2 1 2 2 1 0]
        if self.asyncsq == 0 :
            if iclass!=minclass:
                for q in range(self.neq):
                    for node in interface[iclass-1]:
                        self.async_flux[q,node] += self.beta[irkstep]*field.flux[q][node]
            if iclass!=maxclass:
                for q in range(self.neq):
                    for cell in range(len(adjcells[iclass])):
                        cellid = adjcells[iclass][cell]
                        for node in range(len(interface[iclass])):
                            nodeid = interface[iclass][node]
                            if nodeid == cellid or nodeid == cellid+1:
                                cor_coef = 0.5
                                sign = -1.0
                                if nodeid == cellid:
                                    sign = 1.0
                                field.residual[q][cellid] += sign*(cor_coef*self.async_flux[q,nodeid] - field.flux[q][nodeid])/self.mesh.dx()[cellid]
            #------ clear async flux at [ck,ck+1 boundary] after using it for i=1
                                if irkstep == self.nstage-1:
                                    self.async_flux[q,nodeid] = 0.0 

        #=== CORRECTTION FOR THE SECOND ASYNC ALGO :: field.asyncsq = 2 :=> [0 1 1 2 2 2 2]
        if self.asyncsq == 2 :
            # reset async flux array 
            if iclass == minclass and irkstep == 0 and iasync == 0:
                for q in range(self.neq):
                    for ielem in range(field.nelem+1):
                        self.async_flux[q,ielem] = 0.0 
            if iclass!=maxclass:
                for q in range(self.neq):
                    for node in interface[iclass+1]:
                        self.async_flux[q,node] += self.beta[irkstep]*field.flux[q][node]

            if iclass!=minclass:
                for q in range(self.neq):
                    for cell in range(len(adjcells[iclass])):
                        cellid = adjcells[iclass][cell] 
                        for node in range(len(interface[iclass])):
                            nodeid = interface[iclass][node]
                            if nodeid == cellid or nodeid == cellid+1:
                                cor_coef = 1.0
                                sign = 1.0
                                if nodeid == cellid:
                                    sign = -1.0
                                # cor_coef = 1.0 / (2.0**(iclass - 1))
                                for icoef in range(iclass-1):
                                    cor_coef *= 0.5
                                field.residual[q][cellid] -= sign*(cor_coef*self.async_flux[q,nodeid] - field.flux[q][nodeid])/self.mesh.dx()[cellid]
            #------ clear async flux at [ck,ck+1 boundary] after using it for i=1
                                if irkstep == self.nstage-1 and iasync==len(self.async_seq)-1:
                                    self.async_flux[q][nodeid] = 0.0                             
                                else:
                                    if irkstep == self.nstage-1 and self.async_seq[iasync]!=self.async_seq[iasync+1]:
                                        self.async_flux[q][nodeid] = 0.0 
         
        for q in range(self.neq):
            for i in range(self.nelem):
                if (cell_class[i]==iclass):
                    self.new_rhs[irkstep,q,i] = field.residual[q][i]           
        self.add_res(field,dtloc, irkstep,iclass)
        return field

    def add_res(self,field, dtloc, irkstep,iclass):   
        for rk_coeff_index in range(irkstep+1):
            for i in range(self.neq):
                for icell in range(self.nelem):
                    if (cell_class[icell]==iclass):
                        field.qdata[i][icell] += dtloc * self.RKcoeff[irkstep, rk_coeff_index] * self.new_rhs[rk_coeff_index,i,icell]  # time can be scalar or np.array

#===
#=== Async functions that classify cells
#===

    def cell_classify(self,dt): #dt is the list of all cells time steps    #TB

        dtmin = min(dt) #minimum Dt of all cells
        dtmax = max(dt) #calculated maximum Dt of all cells
        #print "dtmin",dtmin,"dtmax",dtmax

        nc = int(np.log2(dtmax/dtmin))  
        cell_class = np.full(self.nelem,nc,dtype=int) #list of classes per cells initialize as maximum class

        for i in np.arange(self.nelem):
            cell_class[i] = nc-int(np.log2(dt[i]/dtmin))

        next = True

        while next == True:

            next = False

            #Forcing the same class for first and last cells to comply with periodic boundary conditions
            if self.bc == 'p':
                minclass = min(cell_class[0],cell_class[self.nelem-1])
                cell_class[0]            = minclass
                cell_class[self.nelem-1] = minclass

            for i in np.arange(self.nelem-1):

                iclass0 = cell_class[i]   #icv0 = i
                iclass1 = cell_class[i+1] #icv1 = i+1
                cldif   = iclass1-iclass0

                if abs(cldif) > 1:
                    if cldif<0:
                        cell_class[i+1]=iclass0 - 1
                    else:
                        cell_class[i]=iclass1 - 1
                    next = True

            pass

        minclass = min(cell_class)
        maxclass = max(cell_class)

        nc = maxclass-minclass
#        print "number of classes",nc+1

        for i in np.arange(self.nelem):
            cell_class[i] = cell_class[i] - minclass
#        print "cell_class",cell_class

        DT = np.zeros(nc+1)
        for k in np.arange(0,nc+1):
            DT[k] = pow(2,nc-k)*dtmin         #timestep for each class k
#        print "DT",DT

        classes   = [x for x in range(nc+1)]  #list of cells per classes
        adjcells  = [x for x in range(nc+1)]  #list of adjacent cells per classes
        incells   = [x for x in range(nc+1)]  #list of inner    cells per classes
        interface = [x for x in range(nc+1)]  #list of interface nodes per classes

        for k in np.arange(0,nc+1):
            classes[k]   = []
            adjcells[k]  = []
            incells[k]   = []
            interface[k] = []

        #Construct list classes as list (has to be changed if cell_class is modified)
        for i in np.arange(self.nelem):
            for k in np.arange(0,nc+1): #searches to classify cells starting from the coarsest class C0
                if cell_class[i]==k:
                    classes[k].append(i)
#        print "classes",classes

        #Flag adjacent cells as the boundary cells of the coarser class (only one class difference is allowed)
        for i in range(self.nelem-1):

            iclass0 = cell_class[i]   #icv0 = i
            iclass1 = cell_class[i+1] #icv1 = i+1
            cldif   = iclass1-iclass0

            if abs(cldif)==1:                                     #checking when we have a class change
                if self.asyncsq == 0:            
                    if cldif>0:                                #front class > back class
                        if i not in adjcells[cell_class[i]]:      #to avoid double elements
                            adjcells[cell_class[i]].append(i)     #storing current cell i  
                    elif cldif<0:                              #front class < back class
                        if i+1 not in adjcells[cell_class[i+1]]:  #to avoid double elements
                            adjcells[cell_class[i+1]].append(i+1) #storing next cell i+1
                elif self.asyncsq == 2: 
                    if cldif>0:                                #front class > back class
                        if i+1 not in adjcells[cell_class[i+1]]:  #to avoid double elements
                            adjcells[cell_class[i+1]].append(i+1) #storing current cell i  
                    elif cldif<0:                              #front class < back class
                        if i not in adjcells[cell_class[i]]:      #to avoid double elements
                            adjcells[cell_class[i]].append(i)     #storing next cell i+1
            elif abs(cldif)!=0:
                print("there are still adjacent cells of class difference > 1", cldif)
                break
        #print "adjcells",adjcells

        #Flag inner cells as a substraction of two sets(lists)
        for k in np.arange(0,nc+1):  
            incells[k] = list(set(classes[k]) - set(adjcells[k]))
            incells[k].sort()
        #print "incells", incells

        #Flag interface between adjacent cell of class k and inner of class k+1
        for k in np.arange(0,nc+1):
            for c in adjcells[k]:
                if self.asyncsq == 0:
                    if c+1 in incells[k+1]:
                        interface[k].append(c+1)
                    if c-1 in incells[k+1]:
                        interface[k].append(c)
                elif self.asyncsq == 2:
                    if c+1 in incells[k-1]:
                        interface[k].append(c+1)
                    if c-1 in incells[k-1]:
                        interface[k].append(c)
            interface[k].sort()
        #print "interface",interface

        # Computation of Async local time integration sequence
        if self.asyncsq == 0 :# self.asyncsq = 0 :=> [2 2 1 2 2 1 0]
            async_seq = self.asyncseq0(cell_class)
        elif self.asyncsq == 1 :# self.asyncsq = 1 :=> [0 1 2 2 1 2 2]
            async_seq = self.asyncseq1(cell_class)
        elif self.asyncsq == 2 :# self.asyncsq = 2 :=> [0 1 1 2 2 2 2]
            async_seq = self.asyncseq2(cell_class)
        else:
            sys.exit("You entered the wrong index for async seq: 0,1 or 2 !!!!! \n")
        #print async_seq

        return nc, DT, classes, cell_class, adjcells, interface, async_seq
        
    def asyncseq0(self,cell_class):
        
        nmax = max(cell_class)
        nmin = min(cell_class)
        #Total Number of classes
        nclass = nmax - nmin + 1
        #Total Number of Async sequences
        nasync = pow(2,nclass)-1
        #List Array of Async Class Integration sequences
        async_seq = np.zeros(nasync,dtype=np.int8)
        #
        # COMPUTATION OF THE ASYNC CLASS INTEGRATION SEQUENCE LIST
        #
        # Initial first sequences (equivalent to 2 classes)
        indx = pow(2,nclass-1)-1
        if nclass > 1:
            async_seq[indx-1]=1
            async_seq[2*indx-1]=1
        # Fill up the Async Class Integration sequences list array
        for i in range(1,nclass-1):
            nclass_loc = pow(2,nclass-(i+1))-1
            for j in range(nasync):
                if async_seq[j]==i:
                    indx1=j-1
                    indx2=indx1-nclass_loc
                    async_seq[indx1]=i+1
                    async_seq[indx2]=i+1   
        async_seq += nmin
        
        return async_seq

    def asyncseq1(self,cell_class):
        
        nmax = max(cell_class)
        nmin = min(cell_class)
        #Total Number of classes
        nclass = nmax - nmin + 1
        #Total Number of Async sequences
        nasync = pow(2,nclass)-1
        #List Array of Async Class Integration sequences
        async_seq = np.zeros(nasync,dtype=np.int8)
        #
        # COMPUTATION OF THE ASYNC CLASS INTEGRATION SEQUENCE LIST
        #
        # Initial first sequences (equivalent to 2 classes)
        indx = pow(2,nclass-1)-1
        if nclass > 1:
            async_seq[indx-1]=1
            async_seq[2*indx-1]=1
        # Fill up the Async Class Integration sequences list array
        for i in range(1,nclass-1):
            nclass_loc = pow(2,nclass-(i+1))-1
            for j in range(nasync):
                if async_seq[j]==i:
                    indx1=j-1
                    indx2=indx1-nclass_loc
                    async_seq[indx1]=i+1
                    async_seq[indx2]=i+1   
        async_seq += nmin
        async_seq = async_seq[::-1]
        
        return async_seq

    def asyncseq2(self,cell_class):
        
        nmax = max(cell_class)
        nmin = min(cell_class)
        #Total Number of classes
        nclass = nmax - nmin + 1
        #Total Number of Async sequences
        #List Array of Async Class Integration sequences
        async_seq = []
        #
        # COMPUTATION OF THE ASYNC CLASS INTEGRATION SEQUENCE LIST
        #
        # Initial first sequences (equivalent to 2 classes)
        for cl in range(nmin,nmax+1):
            for i in range(2**cl):
                async_seq.append(cl)
        
        async_seq = np.array(async_seq)
        return async_seq

#=== RK Butch arrays 

class AsyncLSrk1(AsyncLowStorageRKmodel):
    def __init__(self, mesh, num):

        self.mesh    = mesh
        self.num     = num
        self.nstage  = 1
        self.RKcoeff = np.array ([[1.]])

        self.beta = np.zeros(self.nstage)
        for i in range(self.nstage):
            for j in range(self.nstage):
                self.beta[j]+=self.RKcoeff[i,j]

class AsyncLSrk22(AsyncLowStorageRKmodel):
    def __init__(self, mesh, num):

        self.mesh    = mesh
        self.num     = num
        self.nstage  = 2
        self.RKcoeff = np.array ([[0.5,  0.],
                                  [-0.5, 1.]])

        self.beta = np.zeros(self.nstage)
        for i in range(self.nstage):
            for j in range(self.nstage):
                self.beta[j]+=self.RKcoeff[i,j]

class AsyncLSrk3ssp(AsyncLowStorageRKmodel):
    def __init__(self, mesh, num):

        self.mesh    = mesh
        self.num     = num
        self.nstage  = 3
        self.RKcoeff = np.array([[    1. ,     0. ,    0.],
                                 [-3./4. ,  1./4. ,    0.],
                                 [-1./12., -1./12., 2./3.]])

        self.beta = np.zeros(self.nstage)
        for i in range(self.nstage):
            for j in range(self.nstage):
                self.beta[j]+=self.RKcoeff[i,j]

class AsyncLSrk3lsw(AsyncLowStorageRKmodel):
    def __init__(self, mesh, num):

        self.mesh    = mesh
        self.num     = num
        self.nstage  = 3
        self.RKcoeff = np.array ([[ 8./15. ,     0. ,    0.],
                                  [-17./60.,  5./12.,    0.],
                                  [      0., -5./12., 3./4.]])

        self.beta = np.zeros(self.nstage)
        for i in range(self.nstage):
            for j in range(self.nstage):
                self.beta[j]+=self.RKcoeff[i,j]

class AsyncLSrk4(AsyncLowStorageRKmodel):
    def __init__(self, mesh, num):

        AsyncLowStorageRKmodel.__init__(self, mesh, num)

        self.mesh    = mesh
        self.num     = num
        self.nstage  = 4
        self.RKcoeff = np.array([[ 1./2.,     0.,     0.,    0.],
                                 [-1./2.,  1./2.,     0.,    0.],
                                 [    0., -1./2.,     1.,    0.],
                                 [ 1./6.,  1./3., -2./3., 1./6.]]) 


        self.beta = np.zeros(self.nstage)
        for i in range(self.nstage):
            for j in range(self.nstage):
                self.beta[j]+=self.RKcoeff[i,j]
