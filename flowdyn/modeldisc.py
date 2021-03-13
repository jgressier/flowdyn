"""
    The ``modeldisc`` module
    =========================
 
    Provides ...
 
    :Example:
 
    >>> import aerokit.aero.Isentropic as Is
    >>> Is.TiTs_Mach(1.)
    1.2
    >>> Is.TiTs_Mach(2., gamma=1.6)
    2.2
 
    Available functions
    -------------------
 
    Provides Ti/Ts Pi/Ps ratios from Mach number and reversed functions.
    Specific heat ratio `gamma` is optionnal and can be specified in the functions itself
    or using aerokit.common.defaultgas module
 """

import numpy               as np
import flowdyn.modelphy.base as model
#import flowdyn.mesh          as mesh
import flowdyn.field         as field

_default_bc = { 'type': 'per' }

class base():
    """
    virtual object `base` which aims at defining R(Q) in dQ/dt = R(Q)
      model : number of equations
      mesh  : mesh
      qdata : list of neq nparray - conservative data 
      pdata : list of neq nparray - primitive    data
      bc    : type of boundary condition - "p"=periodic / "d"=Dirichlet 
    """
    def __init__(self, model, mesh, num, numflux=None, bcL=_default_bc, bcR=_default_bc):
        self.model = model
        self.mesh  = mesh
        self.neq   = model.neq
        self.num     = num
        self.numflux = numflux
        self.nelem = mesh.ncell
        self.time  = 0.
        self.bcL   = bcL
        self.bcR   = bcR
        self.model.initdisc(mesh)

    def copy(self):
        return base(self.model, self.mesh, self.num, self.bcL, self.bcR)

    def fdata(self, data):
        return field.fdata(self.model, self.mesh, data)

    def fdata_fromprim(self, data):
        f = field.fdata(self.model, self.mesh, data)
        return field.fdata(self.model, self.mesh, self.model.prim2cons(f.data))

    def rhs(self, field):
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
    """
    """
    def __init__(self, model, mesh, num, numflux=None, bcL=_default_bc, bcR=_default_bc):
        base.__init__(self, model, mesh, num, numflux)
        self.bcL   = bcL
        self.bcR   = bcR
            
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
            raise NameError("both conditions should be periodic")
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
            raise NameError("both conditions should be periodic")
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
    """
    def __init__(self, model, mesh, num, bclist, numflux=None):
        base.__init__(self, model, mesh, num, numflux)
        self._bclist = bclist
        
        for tag in self.mesh.list_of_bctags():
            if tag not in bclist:
                raise NameError("missing BC tag '"+tag+"' in bclist argument")
            
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
                    self.xgrad[p][:,j*(nx+1)+1:j*(nx+1)+nx]= self.pdata[p][:,j*nx+1:(j+1)*nx]-self.pdata[p][:,nx*j:(j+1)*nx-1]
                    self.ygrad[p][:,nx+j:(nx*nx)+j:nx] = self.pdata[p][:,nx+j:(nx-1)*nx+(j+1):nx]-self.pdata[p][:,j:(nx-1)*nx+j:nx]
            else:
                for j in range(ny):
                    self.xgrad[p][j*(nx+1)+1:j*(nx+1)+nx]= self.pdata[p][j*nx+1:(j+1)*nx]-self.pdata[p][nx*j:(j+1)*nx-1]
                    self.ygrad[p][nx+j:(nx*nx)+j:nx] = self.pdata[p][nx+j:(nx-1)*nx+(j+1):nx]-self.pdata[p][j:(nx-1)*nx+j:nx]
    
    def interp_face(self):
        """
        Computes left and right interpolation to a face, using self (cell) primitive data and (face) gradients
        """
        
        self.pL, self.pR = self.num.interp_face(self.mesh, self.pdata,self.field,self.neq, self.xgrad, self.ygrad)    
       
        #self.pL = self.field.zero_datalist(newdim=self.mesh.nbfaces())
        #self.pR = self.field.zero_datalist(newdim=self.mesh.nbfaces())
        ## reorder data for 1st order extrapolation, except for boundary conditions
        #nx = self.mesh.nx
        #ny = self.mesh.ny
        #fshift = ny*(nx+1)
        #for p in range(self.neq):
        #    if self.pdata[p].ndim == 2:
            ## distribute cell states (by j rows) to i faces
        #        for j in range(ny):
        #            self.pL[p][:,j*(nx+1)+1:(j+1)*(nx+1)] = self.pdata[p][:,j*nx:(j+1)*nx]
        #            self.pR[p][:,j*(nx+1):(j+1)*(nx+1)-1] = self.pdata[p][:,j*nx:(j+1)*nx]
        #        # distribute cell states  (by j rows) to j faces (starting at index ny*(nx+1))
        #        for j in range(ny):
        #            self.pL[p][:,fshift+(j+1)*nx:fshift+(j+2)*nx] = self.pdata[p][:,j*nx:(j+1)*nx]
        #            self.pR[p][:,fshift+ j   *nx:fshift+(j+1)*nx] = self.pdata[p][:,j*nx:(j+1)*nx]
        #    else:
        #        # distribute cell states (by j rows) to i faces
        #        for j in range(ny):
        #            self.pL[p][j*(nx+1)+1:(j+1)*(nx+1)] = self.pdata[p][j*nx:(j+1)*nx]
        #            self.pR[p][j*(nx+1):(j+1)*(nx+1)-1] = self.pdata[p][j*nx:(j+1)*nx]
        #        # distribute cell states  (by j rows) to j faces (starting at index ny*(nx+1))
        #        for j in range(ny):
        #            self.pL[p][fshift+(j+1)*nx:fshift+(j+2)*nx] = self.pdata[p][j*nx:(j+1)*nx]
        #            self.pR[p][fshift+ j   *nx:fshift+(j+1)*nx] = self.pdata[p][j*nx:(j+1)*nx]
    
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
                NameError("unknown face orientation")
            if bcvalue['type'] == 'per':
                conbctag = _connect[bctag]
                # check connected BC is type 'per' too
                if self._bclist[conbctag]['type'] != 'per':
                    raise NameError("both conditions "+bctag+" and "+conbctag+" should be periodic")
                for i in range(self.neq):
                    if self.model.shape[i] == 2: # if i-th data is a vector
                        data_bc[i][:,self.mesh.index_of_bc(bctag)] = data_bc[i][:,self.mesh.index_of_bc(conbctag)]
                    else: # if i-th data is a scalar
                        data_bc[i][self.mesh.index_of_bc(bctag)] = data_bc[i][self.mesh.index_of_bc(conbctag)]
            else: # all other boundary conditions
                dir = self.mesh.normal_of_bc(bctag)
                bcdata_in = [None]*len(data_in)
                iofaces   = self.mesh.index_of_bc(bctag)
                for i,p in enumerate(data_in):
                    if p.ndim == 1:
                        bcdata_in[i] = p[iofaces]
                    elif p.ndim == 2:
                        bcdata_in[i] = p[:,iofaces]
                bcdata_bc = self.model.namedBC(bcvalue['type'], dir, bcdata_in, bcvalue)
                for i,p in enumerate(bcdata_bc):
                        if p.ndim == 1:
                            data_bc[i][iofaces] = p
                        elif p.ndim == 2:
                            data_bc[i][:,iofaces] = p
    
    def calc_bc_grad(self):
        bcper = { 'type': 'per' }
        bclist = self._bclist
        nx = self.mesh.nx
        ny = self.mesh.ny
        if bclist['left'] == bcper and bclist['right'] == bcper:
            for p in range(self.neq):
                if self.pdata[p].ndim == 2:
                    for j in range(ny):
                        self.xgrad[p][:,(nx+1)*j:(nx+1)*(j+1):nx] = 0
                else:
                    for j in range(ny):
                        self.xgrad[p][(nx+1)*j:(nx+1)*(j+1):nx] = 0

        if bclist['top'] == bcper and bclist['bottom'] == bcper:
            for p in range(self.neq):
                if self.pdata[p].ndim == 2:
                    for j in range(nx):
                        self.ygrad[p][:,j] = 0
                        self.ygrad[p][:,ny*ny+j] = 0
                else:
                    for j in range(nx):
                        self.ygrad[p][j] = 0
                        self.ygrad[p][ny*ny+j] = 0

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
        dir = np.zeros((2,nxface+nyface),dtype=np.int8)
        dir[0,:nxface] = 1
        dir[1,nxface:] = 1  
        self.flux = self.model.numflux(self.numflux, self.pL, self.pR, dir) 

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