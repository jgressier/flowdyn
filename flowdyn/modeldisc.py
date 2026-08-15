"""Provide finite-volume spatial discretizations for Flowdyn models."""

import math
from copy import copy as shallow_copy
import numpy               as np
#import flowdyn.modelphy.base as model
#import flowdyn.mesh          as mesh
import flowdyn.field         as field
import flowdyn._data as dd

class base():
    """
    virtual object `base` which aims at defining R(Q) in dQ/dt = R(Q)
      model : number of equations
      mesh  : mesh
      qdata : list of neq nparray - conservative data 
      pdata : list of neq nparray - primitive    data
      bc    : type of boundary condition - "p"=periodic / "d"=Dirichlet 
    """
    def __init__(self, model, mesh, num, numflux=None, bcL=None, bcR=None):
        if not hasattr(num, 'interp_face'):
            raise TypeError("num must provide an interp_face method")
        self.model = model
        self.mesh  = mesh
        self.neq   = model.neq
        self.num     = num
        self.numflux = numflux
        self.nelem = mesh.ncell
        self.time  = 0.
        self.bcL   = self._validated_bc(bcL)
        self.bcR   = self._validated_bc(bcR)
        self.model.initdisc(mesh)

    @staticmethod
    def _validated_bc(bc):
        bc = {'type': 'per'} if bc is None else dict(bc)
        if 'type' not in bc or not isinstance(bc['type'], str):
            raise ValueError("a boundary condition must define a string 'type'")
        return bc

    def copy(self):
        """Create a shallow copy with independent boundary dictionaries."""
        new = shallow_copy(self)
        new.bcL = self.bcL.copy()
        new.bcR = self.bcR.copy()
        return new

    def fdata(self, data):
        return field.fdata(self.model, self.mesh, data)

    def fdata_fromprim(self, data):
        f = field.fdata(self.model, self.mesh, data)
        return field.fdata(self.model, self.mesh, self.model.prim2cons(f.data))

    def average(self, q):
        """compute average of one field only"""
        return self.mesh.average(q)

    def all_L2average(self, qdata):
        """compute average of all fields """
        qavg = [ self.mesh.L2average(dd._vecsqrmag(q)) if q.ndim==2 else self.mesh.L2average(q) for q in qdata ]
        return math.sqrt(np.average(np.square(qavg)))

    def rhs(self, field):
        if field.model is not self.model or field.mesh is not self.mesh:
            raise ValueError("field, model and mesh must match the discretization")
        #print("t=",field.time)
        self.field = field
        self.qdata = [ d.copy() for d in field.data ] # wonder if copy is necessary
        self.cons2prim()
        self.calc_grad()
        self.calc_bc_grad()
        self.interp_face()
        self.calc_bc()
        self.calc_flux()
        self.calc_res()
        if self.model.source: 
            self.add_source()
        return self.residual

    def cons2prim(self):
        self.pdata = self.model.cons2prim(self.qdata)
    
    def prim2cons(self):
        self.qdata = self.model.prim2cons(self.pdata) 
                
# -----------------------------------------------------------------------------------
class fvm1d(base):
    """Implement a one-dimensional finite-volume discretization."""
    def __init__(self, model, mesh, num, numflux=None, bcL=None, bcR=None):
        base.__init__(self, model, mesh, num, numflux, bcL, bcR)
            
    def calc_grad(self):
        """
        Computes face-based gradients of each primitive data
        """
        self.grad = []
        for d in self.pdata:
            g = np.zeros(self.mesh.ncell+1)
            g[1:-1] = (d[1:]-d[0:-1]) / (self.mesh.xc[1:]-self.mesh.xc[0:-1])
            self.grad.append(g)
    
    def interp_face(self):
        """
        Computes left and right interpolation to a face, using self (cell) primitive data and (face) gradients
        """
        self.pL, self.pR = self.num.interp_face(self.mesh, self.pdata, self.grad)             
    
    def calc_bc(self):
        if (self.bcL['type'] == 'per') and (self.bcR['type'] == 'per'):     #periodic boundary conditions
            for i in range(self.neq):
                self.pL[i][0]          = self.pL[i][self.nelem] 
                self.pR[i][self.nelem] = self.pR[i][0] 
        elif (self.bcL['type'] == 'per') or (self.bcR['type'] == 'per'):     # inconsistent periodic boundary conditions:
            raise ValueError("both conditions should be periodic")
        else:
            q_bcL  = self.model.namedBC(self.bcL['type'],
                                        -1, [self.pR[i][0] for i in range(self.neq)], self.bcL)
            q_bcR  = self.model.namedBC(self.bcR['type'],
                                        1, [self.pL[i][self.nelem] for i in range(self.neq)], self.bcR)
            for i in range(self.neq):
                self.pL[i][0]          = q_bcL[i]
                self.pR[i][self.nelem] = q_bcR[i]
    
    def calc_bc_grad(self):
        if (self.bcL['type'] == 'per') and (self.bcR['type'] == 'per'):     #periodic boundary conditions
            for i in range(self.neq):
                self.grad[i][0]  = 0.
                self.grad[i][-1] = 0.
                self.grad[i][0] = self.grad[i][-1] = (self.pdata[i][0]-self.pdata[i][-1]) / (self.mesh.xc[0]+self.mesh.length-self.mesh.xc[-1])
        elif (self.bcL['type'] == 'per') or (self.bcR['type'] == 'per'):     # inconsistent periodic boundary conditions:
            raise ValueError("both conditions should be periodic")
        else:
            for i in range(self.neq):
                self.grad[i][0]  = 0.
                self.grad[i][-1] = 0.
    
    def calc_flux(self):
            self.flux = self.model.numflux(self.numflux, self.pL, self.pR) # get numerical flux from model object, self.numflux is here only a tag

    def calc_timestep(self, f, condition):
        return self.model.timestep(f.data, self.mesh.xf[1:self.nelem+1]-self.mesh.xf[0:self.nelem], condition)
        
    def calc_res(self):
        self.residual = []
        for i in range(self.neq):
            self.residual.append(-(self.flux[i][1:self.nelem+1]-self.flux[i][0:self.nelem]) \
                                  /(self.mesh.dx()))
        return self.residual

    def add_source(self):
        for i in range(self.neq):
            if self.model.source[i]:
                self.residual[i] += self.model.source[i](self.mesh.centers(), self.qdata)
        return self.residual

# -----------------------------------------------------------------------------------
class fvm(fvm1d): # alias fvm->fvm1d for backward compatibility
    pass

# -----------------------------------------------------------------------------------
class fvm2dcart(base):
    """
    2D finite volume discretized operator for cartesian mesh

    internal variables:
        model: physical model
        num: cell to face extrapolation operator
        numflux: numerical flux function
        _bclist: dict of tagged BC and associated type and parameters
        pdata[p]: list of primitive data at cells
        xgrad[p]: list of x difference of primitive data at i-faces only
        ygrad[p]: list of y difference of primitive data at j-faces only

    connectivity:
        cell data are ordered "row wise", same as mesh2d description
            j row: j*nx to (j+1)*nx-1 or (j*nx:(j+1)*nx) python slice
        face data are also ordered "row wise", starting with i faces (nx+1)*ny then j faces nx*(ny+1)
            j row of i-face: j*(nx+1) to (j+1)*(nx+1)-1 or j*(nx+1):(j+1)*(nx+1)-1 python slice
            j row of j-face: j*(nx+1) to (j+1)*(nx+1)-1 or j*(nx+1):(j+1)*(nx+1)-1 python slice (shifted by ny*(nx+1))
    """
    def __init__(self, model, mesh, num, bclist, numflux=None):
        base.__init__(self, model, mesh, num, numflux)
        self._bclist = bclist
        
        for tag in self.mesh.list_of_bctags():
            if tag not in bclist:
                raise ValueError("missing BC tag '"+tag+"' in bclist argument")

    def is_per(self, name):
        return self._bclist[name]['type'] == 'per'

    def calc_grad(self):
        """
        Computes face-based gradients of each primitive data
        """
        self.xgrad=[]
        self.ygrad=[]
        nx = self.mesh.nx
        ny = self.mesh.ny
        self.xgrad = self.field.zero_datalist(newdim=ny*(nx+1))
        self.ygrad = self.field.zero_datalist(newdim=nx*(ny+1))
        for p in range(self.neq):
            if self.pdata[p].ndim == 2:
                for j in range(ny):
                    self.xgrad[p][:,j*(nx+1)+1:j*(nx+1)+nx]= self.pdata[p][:,j*nx+1:(j+1)*nx]-self.pdata[p][:,j*nx:(j+1)*nx-1]
                for j in range(1,ny):
                    self.ygrad[p][:,j*nx:(j+1)*nx] = self.pdata[p][:,j*nx:(j+1)*nx]-self.pdata[p][:,(j-1)*nx:j*nx]
            else:
                for j in range(ny):
                    self.xgrad[p][j*(nx+1)+1:j*(nx+1)+nx]= self.pdata[p][j*nx+1:(j+1)*nx]-self.pdata[p][nx*j:(j+1)*nx-1]
                for j in range(1,ny):
                    self.ygrad[p][j*nx:(j+1)*nx] = self.pdata[p][j*nx:(j+1)*nx]-self.pdata[p][(j-1)*nx:j*nx]
    
    def interp_face(self):
        """
        Computes left and right interpolation to a face, using self (cell) primitive data and (face) gradients
        """ 
        self.pL, self.pR = self.num.interp_face(self.mesh, self.pdata,self.field,self.neq, self.xgrad, self.ygrad)    
    
    def calc_bc(self):
        """
        loop on all bc tags and apply BC from model and list and index 
        """
        _connect = { 'top': 'bottom', 'bottom': 'top', 'right': 'left', 'left': 'right'}
        for bctag, bcvalue in self._bclist.items():
            if self.mesh.bcface_orientation(bctag) == 'inward': # inward faces, L data must be computed
                data_in = self.pR
                data_bc = self.pL
            elif self.mesh.bcface_orientation(bctag) == 'outward': # outward faces, L data must be computed
                data_in = self.pL
                data_bc = self.pR
            else:
                raise ValueError("unknown face orientation")
            if bcvalue['type'] == 'per':
                conbctag = _connect[bctag]
                # check connected BC is type 'per' too
                if self._bclist[conbctag]['type'] != 'per':
                    raise ValueError("both conditions "+bctag+" and "+conbctag+" should be periodic")
                for i in range(self.neq):
                    if self.model.shape[i] == 2: # if i-th data is a vector
                        data_bc[i][:,self.mesh.index_of_bc(bctag)] = data_bc[i][:,self.mesh.index_of_bc(conbctag)]
                    else: # if i-th data is a scalar
                        data_bc[i][self.mesh.index_of_bc(bctag)] = data_bc[i][self.mesh.index_of_bc(conbctag)]
            else: # all other boundary conditions
                direction = self.mesh.normal_of_bc(bctag)
                bcdata_in = [None]*len(data_in)
                iofaces   = self.mesh.index_of_bc(bctag)
                for i,p in enumerate(data_in):
                    if self.model.shape[i] == 1:
                        bcdata_in[i] = p[iofaces]
                    elif self.model.shape[i] == 2:
                        bcdata_in[i] = p[:,iofaces]
                bcdata_bc = self.model.namedBC(bcvalue['type'], direction, bcdata_in, bcvalue)
                for i,p in enumerate(bcdata_bc):
                        if self.model.shape[i] == 1:
                            data_bc[i][iofaces] = p
                        elif self.model.shape[i] == 2:
                            data_bc[i][:,iofaces] = p
    
    def calc_bc_grad(self):
        #bclist = self._bclist
        nx = self.mesh.nx
        ny = self.mesh.ny
        if self.is_per('left') and self.is_per('right'):
            for p in range(self.neq):
                if self.pdata[p].ndim == 2:
                    grad = self.pdata[p][:,::nx]-self.pdata[p][:,nx-1::nx]
                    self.xgrad[p][:,::nx+1] = grad
                    self.xgrad[p][:,nx::nx+1] = grad
                else:
                    grad = self.pdata[p][::nx]-self.pdata[p][nx-1::nx]
                    self.xgrad[p][::nx+1] = grad
                    self.xgrad[p][nx::nx+1] = grad
        elif self.is_per('left') or self.is_per('right'):     # inconsistent periodic boundary conditions:
            raise ValueError("both conditions should be periodic")
        else:
            for p in range(self.neq):
                if self.pdata[p].ndim == 2:
                    self.xgrad[p][:,::nx+1] = 0.
                    self.xgrad[p][:,nx::nx+1] = 0.
                else:
                    self.xgrad[p][::nx+1] = 0.
                    self.xgrad[p][nx::nx+1] = 0.

        if self.is_per('top') and self.is_per('bottom'):
            for p in range(self.neq):
                if self.pdata[p].ndim == 2:
                    grad = self.pdata[p][:,0:nx]-self.pdata[p][:,(ny-1)*nx:]
                    self.ygrad[p][:,0:nx] = grad
                    self.ygrad[p][:,ny*nx:] = grad
                else:
                    grad = self.pdata[p][0:nx]-self.pdata[p][(ny-1)*nx:]
                    self.ygrad[p][0:nx] = grad
                    self.ygrad[p][ny*nx:] = grad
        elif self.is_per('top') or self.is_per('bottom'):     # inconsistent periodic boundary conditions:
            raise ValueError("both conditions should be periodic")
        else:
            for p in range(self.neq):
                if self.pdata[p].ndim == 2:
                    self.ygrad[p][:,0:nx] = 0.
                    self.ygrad[p][:,ny*nx:] = 0.
                else:
                    self.ygrad[p][0:nx] = 0.
                    self.ygrad[p][ny*nx:] = 0.

        return

    def calc_flux(self):
        """
          computes array of fluxes, calls model numerical flux using self.numflux tag
          first (nx+1)*ny are X oriented flux, then nx*(ny+1) are Y oriented flux
        """
        nx = self.mesh.nx
        ny = self.mesh.ny
        nxface = ny*(nx+1)
        nyface = nx*(ny+1)
        # get numerical flux from model object, self.numflux is here only a tag
        direction = np.zeros((2,nxface+nyface),dtype=np.int8)
        direction[0,:nxface] = 1
        direction[1,nxface:] = 1
        self.flux = self.model.numflux(self.numflux, self.pL, self.pR, direction)

    def calc_timestep(self, f, condition):
        # cell characteristic length is constant for cartesian mesh
        dx = self.mesh.dx()
        dy = self.mesh.dy()
        ldim = dx*dy / (dx+dy)
        return self.model.timestep(f.data, ldim, condition)
        
    def calc_res(self):
        self.residual = [ np.zeros_like(d) for d in self.qdata ]
        dx = self.mesh.dx()
        dy = self.mesh.dy()
        nx = self.mesh.nx
        ny = self.mesh.ny
        fshift = ny*(nx+1)
        for flux, res in zip(self.flux, self.residual):
            if flux.ndim == 2:
                # flux balance by j row
                for j in range(ny):
                    res[:,j*nx:(j+1)*nx] -= (flux[:,j*(nx+1)+1:(j+1)*(nx+1)] - flux[:,j*(nx+1):(j+1)*(nx+1)-1] ) /dx + \
                                   (flux[:,fshift+(j+1)*nx:fshift+(j+2)*nx] - flux[:,fshift+j*nx:fshift+(j+1)*nx] ) /dy
            else:
                # flux balance by j row
                for j in range(ny):
                    res[j*nx:(j+1)*nx] -= (flux[j*(nx+1)+1:(j+1)*(nx+1)] - flux[j*(nx+1):(j+1)*(nx+1)-1] ) /dx + \
                            (flux[fshift+(j+1)*nx:fshift+(j+2)*nx] - flux[fshift+j*nx:fshift+(j+1)*nx] ) /dy
        return self.residual

    def add_source(self):
        for i in range(self.neq):
            if self.model.source[i]:
                self.residual[i] += self.model.source[i](self.mesh.centers(), self.qdata)
        return self.residual

# -----------------------------------------------------------------------------------
class fvm2d(fvm2dcart): # alias fvm2d->fvm2dcart
    pass


 
# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()
