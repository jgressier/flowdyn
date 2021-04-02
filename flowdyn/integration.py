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

# from scipy.sparse import csc_matrix
# import scipy.sparse.linalg as splinalg
# from numpy.linalg import inv
import flowdyn.field as field

# --------------------------------------------------------------------
# portage

if float(sys.version[:3]) >= 3.3:  # or 3.8
    myclock = time.process_time
else:
    myclock = time.clock
start = myclock()

# --------------------------------------------------------------------

# --------------------------------------------------------------------
class fakemodel:
    """ """

    def __init__(self):
        self.neq = 1
        self.shape = [1]


class fakemesh:
    """ """

    def __init__(self):
        self.ncell = 1


class fakedisc:
    """ """

    def __init__(self, z):
        self.z = z

    def rhs(self, f):
        """

        Args:
          f:

        Returns:

        """
        return [f.data[0] * self.z]


# --------------------------------------------------------------------
# generic model
# --------------------------------------------------------------------


class timemodel:
    """ """

    def __init__(self, mesh, modeldisc):
        self.mesh = mesh
        self.modeldisc = modeldisc
        self.reset()

    def reset(self):
        """ """
        self._cputime = 0.0
        self._nit = 0

    def calcrhs(self, field):
        """compute RHS with a call to modeldisc function

        Args:
          field:

        Returns: residual/rhs as self.residual
        """
        self.residual = self.modeldisc.rhs(field)

    def step(self, f, dt):
        """virtual method for one step integration

        Args:
          f:
          dt:

        Returns:
        """
        raise NameError("not implemented for virtual class")

    def add_res(self, f, dt, subtimecoef=1.0):
        """

        Args:
          f:
          dt:
          subtimecoef:  (Default value = 1.0)

        Returns:

        """
        f.time += np.min(dt) * subtimecoef
        for i in range(f.neq):
            f.data[i] += dt * self.residual[i]  # time can be scalar or np.array

    def solve(self, f, condition, tsave, flush=None):
        """

        Args:
          f:
          condition:
          tsave:
          flush:  (Default value = None)

        Returns:

        """
        self._nit = 0
        self.condition = condition
        itfield = f.copy()
        if flush:
            alldata = [d for d in itfield.data]
        results = []
        for t in np.arange(len(tsave)):
            endcycle = 0
            while endcycle == 0:
                dtloc = self.modeldisc.calc_timestep(itfield, condition)
                dtloc = min(dtloc)
                if itfield.time + dtloc >= tsave[t]:
                    endcycle = 1
                    dtloc = tsave[t] - itfield.time
                self._nit += 1
                if dtloc > np.spacing(dtloc):
                    self.step(itfield, dtloc)
                if flush:
                    for i, q in zip(range(len(alldata)), itfield.data):
                        alldata[i] = np.vstack((alldata[i], q))
            results.append(itfield.copy())
        self._cputime = myclock() - start
        if flush:
            np.save(flush, alldata)
        return results

    def nit(self):
        """returns number of computed iterations"""
        return self._nit

    def cputime(self):
        """returns cputime"""
        return self._cputime

    def show_perf(self):
        """print performance"""
        print(
            "cpu time computation ({0:d} it) : {1:.3f}s\n  {2:.2f} µs/cell/it".format(
                self._nit,
                self._cputime,
                self._cputime * 1.0e6 / self._nit / self.modeldisc.nelem,
            )
        )

    def propagator(self, z):
        """computes scalar complex propagator of one time step

        Args:
          z:

        Returns:

        """
        # save actual modeldisc
        saved_model = self.modeldisc
        self.modeldisc = fakedisc(z)
        # make virtual field
        f = field.fdata(fakemodel(), fakemesh(), [0 * z + 1.0])
        self.step(f, dt=1.0) # one step with normalized time step
        # get back actual modeldisc
        self.modeldisc = saved_model
        return f.data[0]


# --------------------------------------------------------------------

class explicit(timemodel):
    """ """

    def step(self, field, dtloc):
        """implement 1-step explicit (Euler) method

        Args:
          field: base field for RHS computation
          dtloc: time step, either scalar or array

        Returns: computes RHS and add it to field

        """
        self.calcrhs(field)
        self.add_res(field, dtloc)
        return


class forwardeuler(explicit):  # alias of explicit
    """ """

    pass


# --------------------------------------------------------------------
# RUNGE KUTTA MODELS
# --------------------------------------------------------------------


class rkmodel(timemodel):
    """generic implementation classical Runge-Kutta method
       needs specification of Butcher array from derived class

    Args:

    Returns:

    """
    def __init__(self, mesh, modeldisc):
        timemodel.__init__(self, mesh, modeldisc)
        self.check()

    def check(self):
        """check butcher array and define some algorithm properties"""
        if hasattr(self, '_butcher'):
            self.nstage = len(self._butcher)
            self._subtimecoef = np.zeros(self.nstage)
            for s, pcoef in enumerate(self._butcher):
                self._subtimecoef[s] = np.sum(pcoef)
        else:
            raise NameError("bad implementation of RK model in "+self.__class__.__name__+": Butcher array is missing")

    def step(self, field, dtloc):
        """

        Args:
          field:
          dtloc:

        Returns:
        """
        prhs = []
        pfield = field.copy()
        for s, pcoef in enumerate(self._butcher):
            # compute residual of previous stage and memorize it in prhs[]
            self.calcrhs(pfield)  # result in self.residual
            prhs.append([q.copy() for q in self.residual])
            # revert to initial step
            pfield = field.copy()
            # aggregate residuals
            for qf in self.residual: # multiply last residual first ...
                qf *= pcoef[-1]
            for i in range(pcoef.size - 1):
                for q in range(pfield.neq):
                    self.residual[q] += pcoef[i] * prhs[i][q] # ... and add previous RHS
            # substep
            self.add_res(pfield, dtloc, self._subtimecoef[s])
        field.set(pfield)
        return


class rk2(timemodel):
    """ """

    def step(self, field, dtloc):
        """

        Args:
          field:
          dtloc:

        Returns:

        """
        pfield = field.copy()
        self.calcrhs(pfield)
        self.add_res(pfield, dtloc / 2)
        self.calcrhs(pfield)
        # self.residual = field.residual
        self.add_res(field, dtloc)
        return


class rk3ssp(rkmodel):
    """3rd order RK model with (SSP) Strong Stability Preserving"""
    _butcher = [
        np.array([1.0]),
        np.array([0.25, 0.25]),
        np.array([1.0, 1.0, 4.0]) / 6.0  ]


class rk4(rkmodel):
    """Classical 4th order RK"""
    _butcher = [
            np.array([0.5]),
            np.array([0.0, 0.5]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 2.0, 2.0, 1.0]) / 6.0  ]

class rk2_heun(rkmodel):
    """RK 2nd order Heun's method (or trapezoidal)"""
    _butcher = [np.array([1.0]), np.array([0.5, 0.5])]

class rk3_heun(rkmodel):
    """RK 3rd order Heun's method """
    _butcher = [
            np.array([1.0 / 3.0]),
            np.array([0, 2.0 / 3.0]),
            np.array([0.25, 0, 0.75])     ]


# --------------------------------------------------------------------
# LOW STORAGE RUNGE KUTTA MODELS
# --------------------------------------------------------------------

class LSrkmodelHH(rkmodel):
    """generic implementation of LOW-STORAGE Runge-Kutta method

    Hu and Hussaini (JCP, 1996) method needs p-1 coefficients (_beta)
    needs specification of Butcher array from derived class
        $ for 1<=s<=p, Qs = Q0 + dt * _beta_s RHS(Q_{s-1}) $

    Args:

    Returns:

    """
    # def __init__(self, mesh, modeldisc):
    #     timemodel.__init__(self, mesh, modeldisc)
    #     self.check()

    def check(self):
        """check butcher array and define some algorithm properties"""
        if hasattr(self, '_beta'):
            self.nstage = len(self._beta)
            self._subtimecoef = self._beta
        else:
            raise NameError("bad implementation of RK model in "+self.__class__.__name__+": LSRK array is missing")

    def step(self, field, dtloc):
        """

        Args:
          field:
          dtloc:

        Returns:
        """
        pfield = field.copy()
        for beta in self._beta:
            # compute residual of previous stage and memorize it in prhs[]
            self.calcrhs(pfield)  # result in self.residual
            # substep
            pfield = field.copy()
            self.add_res(pfield, dtloc*beta, beta) # beta is the subtimecoef
        field.set(pfield)
        return

class lsrk25bb(LSrkmodelHH):
    """Low Storage implementation of Bogey Bailly (JCP 2004) 2nd order 5 stages Runge Kutta """
    _beta = [ 0.1815754863270908, 0.238260222208392, 0.330500707328, 0.5, 1. ]

class lsrk26bb(LSrkmodelHH):
    """Low Storage implementation of Bogey Bailly (JCP 2004) 2nd order 6 stages Runge Kutta """
    _beta = [  0.11797990162882 , 0.18464696649448 , 0.24662360430959 , 0.33183954253762 , 0.5, 1. ]

class lsrk4(LSrkmodelHH):
    """RK4 to check"""
    _beta = [ 1./4. , 1./3. , 0.5, 1. ]

# --------------------------------------------------------------------
# IMPLICIT MODELS
# --------------------------------------------------------------------


class implicitmodel(timemodel):
    """generic class for implicit models
    needs specific implementation of step method for derived classes
    TODO: define keywords for inversion method
    TODO: define options, maxit, residuals, save local convergence, condition number

    Args:

    Returns:

    """

    def step(self, field, dtloc):
        """

        Args:
          field:
          dtloc:

        Returns:

        """
        print("not implemented for virtual implicit class")

    def calc_jacobian(self, field, epsdiff=1.0e-6):
        """jacobian matrix dR/dQ of dQ/dt=R(Q) is computed as successive columns by finite difference of R(Q+dQ)
            ordering is ncell x neq (neq is the fast index)

        Args:
          field:
          epsdiff:  (Default value = 1.e-6)

        Returns:

        """
        if (field.model.islinear == 1) and (hasattr(self, "jacobian_use")):
            return
        self.neq = field.neq
        self.dim = self.neq * field.nelem
        self.jacobian = np.zeros([self.dim, self.dim])
        eps = [
            epsdiff * math.sqrt(np.spacing(1.0)) * np.sum(np.abs(q)) / field.nelem
            for q in field.data
        ]
        self.calcrhs(field)
        refrhs = [qf.copy() for qf in self.residual]
        for i in range(field.nelem):  # for all variables (nelem*neq)
            for q in range(self.neq):
                dfield = field.copy()
                dfield.data[q][i] += eps[q]
                self.calcrhs(dfield)
                drhs = [qf.copy() for qf in self.residual]
                for qq in range(self.neq):
                    self.jacobian[qq :: self.neq, i * self.neq + q] = (
                        drhs[qq] - refrhs[qq]
                    ) / eps[q]
        self.jacobian_use = 0
        return self.jacobian

    def solve_implicit(self, field, dtloc, invertion=np.linalg.solve, theta=1.0, xi=0):
        """

        Args:
          field:
          dtloc:
          invertion:  (Default value = np.linalg.solve)
          theta:  (Default value = 1.)
          xi:  (Default value = 0)

        Returns:

        """
        ""
        diag = np.repeat(
            np.ones(field.nelem) / dtloc, self.neq
        )  # dtloc can be scalar or np.array, neq is the fast index
        mat = (1 + xi) * np.diag(diag) - theta * self.jacobian
        rhs = np.zeros((self.dim))
        for q in range(self.neq):
            rhs[q :: self.neq] = self.residual[q]
        if xi != 0:
            for q in range(self.neq):
                rhs[q :: self.neq] += xi * self._lastresidual[q]
        newrhs = invertion(mat, rhs)
        # may change diagonal of mat to avoid division by dtloc
        self.residual = [newrhs[iq :: self.neq] / dtloc for iq in range(self.neq)]


class implicit(implicitmodel):
    """make an Euler implicit or backward Euler step: Qn+1 - Qn = Rn+1"""

    def step(self, field, dtloc):
        """

        Args:
          field:
          dtloc:

        Returns:

        """
        self.calc_jacobian(field)
        self.calcrhs(field)  # compute and define self.residual
        self.solve_implicit(field, dtloc)  # save self.residual
        self.add_res(field, dtloc)
        return


class backwardeuler(implicit):
    """ """

    pass


class trapezoidal(implicitmodel):
    """make an 2nd order (centered) Crank-Nicolson step: Qn+1 - Qn = .5*(Rn + Rn+1)"""

    def step(self, field, dtloc):
        """

        Args:
          field:
          dtloc:

        Returns:

        """
        self.calc_jacobian(field)
        self.calcrhs(field)
        self.solve_implicit(field, dtloc, theta=0.5)
        self.add_res(field, dtloc)
        return


class cranknicolson(trapezoidal):
    """ """
    pass


class gear(trapezoidal):
    """make an 2nd order backward step (Gear): (3Qn+1 - 4Qn + Qn-1)/3 = 2/3* Rn+1
        Using Rn+1 = Rn + A*(Qn+1-Qn), linearized form is (3I-2A)(Qn+1-Qn)=2Rn+(Qn-Qn-1)

    Args:

    Returns:

    """

    def step(self, field, dtloc):
        """

        Args:
          field:
          dtloc:

        Returns:

        """
        if not hasattr(
            self, "_lastresidual"
        ):  # if starting integration (missing last residual), so use 2nd order trapezoidal/cranknicolson
            trapezoidal.step(self, field, dtloc)
            self.add_res(field, dtloc)
        else:
            self.calc_jacobian(field)
            self.calcrhs(field)
            self.solve_implicit(field, dtloc, theta=1.0, xi=0.5)
            self.add_res(field, dtloc)
        self._lastresidual = self.residual
        return




# class LSrk3lsw(LowStorageRKmodel):
#     """ """

#     def __init__(self, mesh, num):

#         self.mesh = mesh
#         self.num = num
#         self.nstage = 3
#         self.RKcoeff = np.array(
#             [
#                 [8.0 / 15.0, 0.0, 0.0],
#                 [-17.0 / 60.0, 5.0 / 12.0, 0.0],
#                 [0.0, -5.0 / 12.0, 3.0 / 4.0],
#             ]
#         )


# --------------------------------------------------------------------
# for tests

List_LSRK_Integrators = [ lsrk25bb, lsrk26bb ]
List_RK_Integrators = [ rk2, rk2_heun, rk3_heun, rk3ssp, rk4 ] + List_LSRK_Integrators
List_Explicit_Integrators = [ explicit ] + List_RK_Integrators
List_Implicit_Integrators = [ implicit, cranknicolson, gear ]
