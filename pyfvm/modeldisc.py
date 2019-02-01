"""
    The ``modeldisc`` module
    =========================
 
    Provides ...
 
    :Example:
 
    >>> import hades.aero.Isentropic as Is
    >>> Is.TiTs_Mach(1.)
    1.2
    >>> Is.TiTs_Mach(2., gamma=1.6)
    2.2
 
    Available functions
    -------------------
 
    Provides Ti/Ts Pi/Ps ratios from Mach number and reversed functions.
    Specific heat ratio `gamma` is optionnal and can be specified in the functions itself
    or using hades.common.defaultgas module
 """


import numpy               as np
import pyfvm.modelphy.base as model
import pyfvm.mesh          as mesh
import pyfvm.field         as field

class base():
    """
    virtual object `base` which aims at defining R(Q) in dQ/dt = R(Q)
      model : number of equations
      mesh  : mesh
      qdata : list of neq nparray - conservative data 
      pdata : list of neq nparray - primitive    data
      bc    : type of boundary condition - "p"=periodic / "d"=Dirichlet 
    """
    def __init__(self, model, mesh, num, bcL='per', bcR='per'):
        self.model = model
        self.mesh  = mesh
        self.neq   = model.neq
        self.num   = num
        self.nelem = mesh.ncell
        self.time  = 0.
        self.bcL   = bcL
        self.bcR   = bcR
        self.model.initdisc(mesh)

    def copy(self):
        return base(self.model, self.mesh, self.num, self.bc, self.bcvalues)

    def fdata(self, data):
        return field.fdata(self.model, self.mesh, data)

    def rhs(self, field):
        print "not implemented for virtual class"


class fvm(base):
    """
    define field: neq x nelem data
      model : number of equations
      nelem : number of cells (conservative and primitive data)
      qdata : list of neq nparray - conservative data 
      pdata : list of neq nparray - primitive    data
      bc    : type of boundary condition - "p"=periodic / "d"=Dirichlet 
    """

    def rhs(self, field):
        self.qdata = [ d.copy() for d in field.data ]
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
                
    def calc_grad(self):
        self.grad = []
        for d in self.pdata:
            g = np.zeros(self.mesh.ncell+1)
            g[1:-1] = (d[1:]-d[0:-1]) / (self.mesh.xc[1:]-self.mesh.xc[0:-1])
            self.grad.append(g)
    
    def interp_face(self):
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
        for i in range(self.neq):
            self.grad[i][0] = self.grad[i][-1] = (self.pdata[i][0]-self.pdata[i][-1]) / (self.mesh.xc[0]+self.mesh.length-self.mesh.xc[-1])
            #print 'BC L/R',self.pL[i], self.pR[i]
    
    def calc_flux(self):
            self.flux = self.model.numflux(self.pL, self.pR)

    def calc_timestep(self, f, condition):
        return self.model.timestep(f.data, self.mesh.xf[1:self.nelem+1]-self.mesh.xf[0:self.nelem], condition)
        
    def calc_res(self):
        self.residual = []
        for i in range(self.neq):
            self.residual.append(-(self.flux[i][1:self.nelem+1]-self.flux[i][0:self.nelem]) \
                                  /(self.mesh.xf[1:self.nelem+1]-self.mesh.xf[0:self.nelem]))
        return self.residual

    def add_source(self):
        for i in range(self.neq):
            if self.model.source[i]:
                self.residual[i] += self.model.source[i](self.mesh.centers(), self.qdata)
        return self.residual

# class scafield(field):
#     def __init__(self, model, bc, nelem=100, bcvalues = []):
#         field.__init__(self, model, bc, nelem=nelem, bcvalues=bcvalues)
            
#     def scadata(self):
#         return self.data[0]


 
# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()