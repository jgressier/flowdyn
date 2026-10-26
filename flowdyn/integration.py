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
import sys
import time

import numpy as np
from scipy.optimize import newton

# from scipy.sparse import csc_matrix
# import scipy.sparse.linalg as splinalg
# from numpy.linalg import inv
import flowdyn.field as field
from flowdyn.monitors import monitor

# --------------------------------------------------------------------
# portage

# if float(sys.version[:3]) >= 3.3:  # or 3.8 !!! fail test of 3.10
#     myclock = time.process_time
# else:
#     myclock = time.clock
myclock = time.process_time  # remove test since minimum version is 3.7

# --------------------------------------------------------------------


class fakemodel:
    """Provide the minimal model interface used by stability tests."""

    def __init__(self):
        self.neq = 1
        self.shape = [1]


class fakemesh:
    """Provide the minimal mesh interface used by stability tests."""

    def __init__(self):
        self.ncell = 1


class fakedisc:
    """Provide a scalar discretization used to evaluate propagators."""

    def __init__(self, z):
        self.z = z

    def rhs(self, f):
        """Compute the scalar right-hand side for a test field.

        Args:
            f: Field for which to compute the right-hand side.

        Returns:
            A one-component residual list.
        """
        return [f.data[0] * self.z]


# --------------------------------------------------------------------
# generic model
# --------------------------------------------------------------------
class _coreiterative:

    def __init__(self):
        self.reset()

    def reset(self, itstart=0):
        """Reset iteration counters and accumulated CPU time."""
        self._cputime = 0.0
        self._nit = 0
        self._itstart = itstart

    def totnit(self):
        """returns total number of computed iterations"""
        return self._itstart + self._nit

    def nit(self):
        """returns number of computed iterations"""
        return self._nit

    def cputime(self):
        """returns cputime"""
        return self._cputime

    def perf_micros(self):
        """returns perf in µs"""
        return self.cputime() * 1.0e6 / self.nit() / self.modeldisc.nelem

    def show_perf(self):
        """print performance"""
        print(
            "cpu time computation ({0:d} it) : {1:.3f}s\n  {2:.2f} µs/cell/it".format(
                self._nit,
                self._cputime,
                self.perf_micros(),
            )
        )


class timemodel(_coreiterative):
    """Provide common services for explicit and implicit time integrators."""

    __default_monitor_freq = 10

    def __init__(self, mesh, modeldisc, monitors=None):
        """Initialize a time integrator for a mesh and spatial discretization."""
        _coreiterative.__init__(self)
        self.mesh = mesh
        self.modeldisc = modeldisc
        self.monitors = {} if monitors is None else monitors
        # define function for monitoring
        self._monitordict = {'residual': self.mon_residual, 'data_average': self.mon_dataavg}

    def calcrhs(self, field):
        """Compute and store the spatial residual.

        Args:
            field: Field for which to compute the residual.
        """
        self.residual = self.modeldisc.rhs(field)

    def step(self, f, dtloc):
        """Advance a field by one time step.

        Args:
            f: Field to update in place.
            dtloc: Scalar or local time-step array.

        Raises:
            NotImplementedError: Always, unless a derived integrator implements the method.
        """
        raise NotImplementedError("step must be implemented by a time integrator")

    def add_res(self, f, dt, subtimecoef=1.0):
        """Apply the current residual to a field.

        Args:
            f: Field to update in place.
            dt: Scalar or local time-step array.
            subtimecoef: Fraction of the time step represented by this stage.
        """
        f.time += np.min(dt) * subtimecoef
        for i in range(f.neq):
            f.data[i] += dt * self.residual[i]  # time can be scalar or np.array

    def _check_end(self, stop):
        """Return whether any configured stopping criterion has been reached."""
        check_end = {}
        for key, value in stop.items():
            if key == 'tottime':
                check_end[key] = self._time >= value
            if key == 'maxit':
                check_end[key] = self._nit >= value
        return any(check_end.values())

    def _parse_monitors(self, monitors):
        """Parse dictionnary of monitors and apply associated function"""
        for name, monval in monitors.items():
            montype = monval.get('type', name)  # if type not set, name can be the type
            if montype in self._monitordict.keys():
                self._monitordict[montype](monval)
            else:
                raise ValueError("unknown monitor type: " + montype)

    def _remove_monitor_output(self, monitors):
        """Parse dictionnary of monitors and apply associated function"""
        for key in monitors.keys():
            monitors[key].pop("output", None)  # None is needed to prevent missing key error

    def solve_legacy(self, f, condition, tsave, stop=None, flush=None, monitors=None):
        """Integrate a field using the legacy solver loop.

        Args:
            f: Initial field.
            condition: CFL number.
            tsave: Times at which to save a solution.
            stop: Optional stopping criteria.
            flush: Optional path for flushed solution data.
            monitors: Optional monitor directives.

        Returns:
            Solution fields corresponding to ``tsave``.
        """
        monitors = {} if monitors is None else monitors
        self.reset()  # reset cputime and nit
        self.condition = condition
        # initialization before loop
        itfield = f.copy()
        if flush:
            alldata = [d for d in itfield.data]
        results = []
        start = myclock()
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

    def solve(self, f, condition, tsave=None, stop=None, flush=None, monitors=None, directives=None):
        """Integrate a field and return solutions at the requested times."""
        tsave = [] if tsave is None else tsave
        monitors = {} if monitors is None else monitors
        directives = {} if directives is None else directives
        self.reset(itstart=0)  # reset cputime and nit
        self._remove_monitor_output(monitors)
        return self._solve(f, condition, tsave, stop, flush, monitors, directives)

    def restart(self, f, condition, tsave=None, stop=None, flush=None, monitors=None, directives=None):
        """Restart integration from an existing field."""
        tsave = [] if tsave is None else tsave
        monitors = {} if monitors is None else monitors
        directives = {} if directives is None else directives
        self.reset(itstart=max(f.it, 0))  # reset cputime and nit
        return self._solve(f, condition, tsave, stop, flush, monitors, directives)

    @staticmethod
    def _validated_solve_inputs(condition, tsave, stop):
        """Validate solver inputs and return normalized save times."""
        if not np.isscalar(condition) or not np.isfinite(condition) or condition <= 0.0:
            raise ValueError("condition must be a positive finite scalar")
        tsave = np.asarray(tsave, dtype=float)
        if np.any(~np.isfinite(tsave)) or np.any(np.diff(tsave) < 0.0):
            raise ValueError("tsave must contain finite, non-decreasing times")
        if stop is not None:
            unknown = set(stop) - {'tottime', 'maxit'}
            if unknown:
                raise ValueError(f"unknown stopping criteria: {', '.join(sorted(unknown))}")
        return tsave

    @staticmethod
    def _stopping_criteria(tsave, stop):
        """Combine the final save time with explicit stopping criteria."""
        criteria = {'tottime': tsave[-1]} if len(tsave) > 0 else {}
        if stop is not None:
            criteria.update(stop)
        if not criteria:
            raise ValueError("missing stopping criteria")
        return criteria

    @staticmethod
    def _first_save_index(current_time, tsave):
        """Return the first requested save time not preceding the current state."""
        return int(np.searchsorted(tsave, current_time, side='left'))

    def _save_if_due(self, results, isave, tsave, mindtloc, verbose):
        """Save an interpolated integration state when the next step crosses a save time."""
        if isave >= len(tsave) or self.Qn.time + mindtloc < tsave[isave]:
            return isave
        saved = self.Qn.copy()
        self.step(saved, tsave[isave] - self.Qn.time)
        saved.it = self._itstart + self._nit
        results.append(saved)
        if verbose:
            print("save state at it {:5d} and time {:6.2e}".format(self._nit, saved.time))
        return isave + 1

    @staticmethod
    def _append_flush_data(alldata, qdata):
        """Append one state to arrays accumulated for file output."""
        for index, values in enumerate(qdata):
            alldata[index] = np.vstack((alldata[index], values))

    def _solve(self, f, condition, tsave, stop, flush, monitors, directives):
        """Integrate a field using the configured time-stepping method.

        Args:
            f: Initial field.
            condition: CFL number.
            tsave: Times at which to save a solution.
            stop: Optional stopping criteria.
            flush: Optional path for flushed solution data.
            monitors: Monitor directives.
            directives: Solver-control directives.

        Returns:
            Solution fields corresponding to ``tsave``.
        """
        tsave = self._validated_solve_inputs(condition, tsave, stop)
        self._time = f.time
        verbose = 'verbose' in directives
        dtlocal = 'dtlocal' in directives
        if verbose and dtlocal:
            print("- dtlocal on")
        self.condition = condition
        stopcrit = self._stopping_criteria(tsave, stop)
        monitors = {**self.monitors, **monitors}
        self.Qn = f.copy()
        alldata = [data for data in self.Qn.data] if flush else None
        results = field.fieldlist()
        start = myclock()
        isave = self._first_save_index(self.Qn.time, tsave)
        advanced = False
        self._parse_monitors(monitors)

        while not self._check_end(stopcrit):
            dtloc = self.modeldisc.calc_timestep(self.Qn, condition)
            mindtloc = min(dtloc)
            isave = self._save_if_due(results, isave, tsave, mindtloc, verbose)
            next_state = self.Qn.copy()
            self.step(next_state, dtloc if dtlocal else mindtloc)
            self.Qn = next_state
            advanced = True
            self._nit += 1
            self._time = self.Qn.time
            self._parse_monitors(monitors)
            if flush:
                self._append_flush_data(alldata, self.Qn.data)

        if advanced and len(results) == 0:
            results.append(self.Qn)
        self._cputime = myclock() - start
        if flush:
            np.save(flush, alldata)
        return results

    def mon_residual(self, params: dict):
        """compute residual average and monitor it

        Args:
            params (dict): [description]
        """
        if self.totnit() % params.get('frequency', self.__default_monitor_freq) == 0:
            if 'output' not in params:
                params['output'] = monitor('residual')
            mon = params['output']
            self.calcrhs(self.Qn)
            value = self.modeldisc.all_L2average(self.residual)
            mon.append(it=self.totnit(), time=self._time, value=value)

    def mon_dataavg(self, params: dict):
        """compute average of current field and monitor it

        Args:
            params (dict): [description]
        """
        if self.totnit() % params.get('frequency', self.__default_monitor_freq) == 0:
            if 'output' not in params:
                params['output'] = monitor('data_average')
            mon = params['output']
            value = self.Qn.average(params['data'])
            mon.append(it=self.totnit(), time=self._time, value=value)

    def propagator(self, z):
        """Compute the scalar complex propagator of one time step.

        Args:
            z: Complex stability-plane coordinate.

        Returns:
            The scalar amplification factor.
        """
        # save actual modeldisc
        saved_model = self.modeldisc
        self.modeldisc = fakedisc(z)
        # make virtual field
        f = field.fdata(fakemodel(), fakemesh(), [0 * z + 1.0])
        self.step(f, dtloc=1.0)  # one step with normalized time step
        # get back actual modeldisc
        self.modeldisc = saved_model
        return f.data[0]

    def cflmax(self):
        """estimation of maximum cfl, may not converge"""

        def _gain_imag(sigma):
            return abs(self.propagator(1j * sigma)) - 1.0

        return newton(_gain_imag, 10.0)


# --------------------------------------------------------------------


class explicit(timemodel):
    """Implement the forward-Euler time integrator."""

    def step(self, field, dtloc):
        """Advance a field by one forward-Euler step.

        Args:
            field: Field to update in place.
            dtloc: Scalar or local time-step array.
        """
        self.calcrhs(field)
        self.add_res(field, dtloc)
        return


class forwardeuler(explicit):  # alias of explicit
    """Provide a descriptive alias for the explicit Euler integrator."""

    pass


# --------------------------------------------------------------------
# RUNGE KUTTA MODELS
# --------------------------------------------------------------------


class rkmodel(timemodel):
    """Implement a generic Runge-Kutta method.

    Derived classes must define a Butcher array.
    """

    def __init__(self, mesh, modeldisc, monitors=None):
        timemodel.__init__(self, mesh, modeldisc, monitors)
        self.check()

    def check(self):
        """check butcher array and define some algorithm properties"""
        if hasattr(self, '_butcher'):
            self.nstage = len(self._butcher)
            self._subtimecoef = np.zeros(self.nstage)
            for s, pcoef in enumerate(self._butcher):
                self._subtimecoef[s] = np.sum(pcoef)
        else:
            raise TypeError(
                "bad implementation of RK model in " + self.__class__.__name__ + ": Butcher array is missing"
            )

    def step(self, field, dtloc):
        """Advance a field by one Runge-Kutta time step.

        Args:
            field: Field to update in place.
            dtloc: Scalar or local time-step array.
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
            for qf in self.residual:  # multiply last residual first ...
                qf *= pcoef[-1]
            for i in range(pcoef.size - 1):
                for q in range(pfield.neq):
                    self.residual[q] += pcoef[i] * prhs[i][q]  # ... and add previous RHS
            # substep
            self.add_res(pfield, dtloc, self._subtimecoef[s])
        field.set(pfield)
        return


class rk2(timemodel):
    """Implement the second-order Runge-Kutta method."""

    def step(self, field, dtloc):
        """Advance a field by one second-order Runge-Kutta step.

        Args:
            field: Field to update in place.
            dtloc: Scalar or local time-step array.
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

    _butcher = [np.array([1.0]), np.array([0.25, 0.25]), np.array([1.0, 1.0, 4.0]) / 6.0]


class rk4(rkmodel):
    """Classical 4th order RK"""

    _butcher = [
        np.array([0.5]),
        np.array([0.0, 0.5]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 2.0, 2.0, 1.0]) / 6.0,
    ]


class rk2_heun(rkmodel):
    """RK 2nd order Heun's method (or trapezoidal)"""

    _butcher = [np.array([1.0]), np.array([0.5, 0.5])]


class rk3_heun(rkmodel):
    """RK 3rd order Heun's method"""

    _butcher = [np.array([1.0 / 3.0]), np.array([0, 2.0 / 3.0]), np.array([0.25, 0, 0.75])]


# --------------------------------------------------------------------
# LOW STORAGE RUNGE KUTTA MODELS
# --------------------------------------------------------------------


class LSrkmodelHH(timemodel):
    """Implement the Hu-Hussaini low-storage Runge-Kutta method.

    Derived classes must provide the ``_beta`` coefficients.
    """

    def __init__(self, mesh, modeldisc, monitors=None):
        timemodel.__init__(self, mesh, modeldisc, monitors)
        self.check()

    def check(self):
        """check butcher array and define some algorithm properties"""
        if hasattr(self, '_beta'):
            self.nstage = len(self._beta)
            self._subtimecoef = self._beta
        else:
            raise TypeError(
                "bad implementation of RK model in " + self.__class__.__name__ + ": LSRK array is missing"
            )

    def step(self, field, dtloc):
        """Advance a field by one low-storage Runge-Kutta step.

        Args:
            field: Field to update in place.
            dtloc: Scalar or local time-step array.
        """
        pfield = field.copy()
        for beta in self._beta:
            # compute residual of previous stage and memorize it in prhs[]
            self.calcrhs(pfield)  # result in self.residual
            # substep
            pfield = field.copy()
            self.add_res(pfield, dtloc * beta, beta)  # beta is the subtimecoef
        field.set(pfield)
        return


class lsrk25bb(LSrkmodelHH):
    """Low Storage implementation of Bogey Bailly (JCP 2004) 2nd order 5 stages Runge Kutta"""

    _beta = [0.1815754863270908, 0.238260222208392, 0.330500707328, 0.5, 1.0]


class lsrk26bb(LSrkmodelHH):
    """Low Storage implementation of Bogey Bailly (JCP 2004) 2nd order 6 stages Runge Kutta"""

    _beta = [0.11797990162882, 0.18464696649448, 0.24662360430959, 0.33183954253762, 0.5, 1.0]


class lsrk4(LSrkmodelHH):
    """RK4 to check"""

    _beta = [1.0 / 4.0, 1.0 / 3.0, 0.5, 1.0]


# --------------------------------------------------------------------
# IMPLICIT MODELS
# --------------------------------------------------------------------


class implicitmodel(timemodel):
    """Provide common operations for implicit time integrators.

    Derived classes must implement the time-stepping method.
    """

    def step(self, field, dtloc):
        """Advance a field by one implicit time step.

        Args:
            field: Field to update in place.
            dtloc: Scalar or local time-step array.

        Raises:
            NotImplementedError: Always, unless a derived class implements the method.
        """
        raise NotImplementedError("not implemented: virtual implicit class")

    def calc_jacobian(self, field, epsdiff=1.0e-6):
        """Compute the residual Jacobian using finite differences.

        Args:
            field: Field at which to evaluate the Jacobian.
            epsdiff: Relative finite-difference perturbation.

        Returns:
            The Jacobian matrix, ordered with the equation index varying fastest.
        """
        if (field.model.islinear == 1) and (hasattr(self, "jacobian_use")):
            return
        self.neq = field.neq
        self.dim = self.neq * field.nelem
        self.jacobian = np.zeros([self.dim, self.dim])
        eps = [epsdiff * math.sqrt(np.spacing(1.0)) * np.sum(np.abs(q)) / field.nelem for q in field.data]
        self.calcrhs(field)
        refrhs = [qf.copy() for qf in self.residual]
        for i in range(field.nelem):  # for all variables (nelem*neq)
            for q in range(self.neq):
                dfield = field.copy()
                dfield.data[q][i] += eps[q]
                self.calcrhs(dfield)
                drhs = [qf.copy() for qf in self.residual]
                for qq in range(self.neq):
                    self.jacobian[qq :: self.neq, i * self.neq + q] = (drhs[qq] - refrhs[qq]) / eps[q]
        self.jacobian_use = 0
        return self.jacobian

    def solve_implicit(self, field, dtloc, invertion=np.linalg.solve, theta=1.0, xi=0):
        """Solve the linearized implicit update system.

        Args:
            field: Field associated with the current residual.
            dtloc: Scalar or local time-step array.
            invertion: Linear-system solver.
            theta: Weight applied to the current Jacobian.
            xi: Weight applied to the previous residual.
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
        """Advance a field by one backward-Euler step.

        Args:
            field: Field to update in place.
            dtloc: Scalar or local time-step array.
        """
        self.calc_jacobian(field)
        self.calcrhs(field)  # compute and define self.residual
        self.solve_implicit(field, dtloc)  # save self.residual
        self.add_res(field, dtloc)
        return


class backwardeuler(implicit):
    """Provide a descriptive alias for the implicit Euler integrator."""

    pass


class trapezoidal(implicitmodel):
    """make an 2nd order (centered) Crank-Nicolson step: Qn+1 - Qn = .5*(Rn + Rn+1)"""

    def step(self, field, dtloc):
        """Advance a field by one trapezoidal step.

        Args:
            field: Field to update in place.
            dtloc: Scalar or local time-step array.
        """
        self.calc_jacobian(field)
        self.calcrhs(field)
        self.solve_implicit(field, dtloc, theta=0.5)
        self.add_res(field, dtloc)
        return


class cranknicolson(trapezoidal):
    """Provide a descriptive alias for the trapezoidal integrator."""

    pass


class gear(trapezoidal):
    """Implement the second-order backward differentiation formula.

    The method uses ``(3 Qn+1 - 4 Qn + Qn-1) / 3 = 2 Rn+1 / 3``.
    """

    def step(self, field, dtloc):
        """Advance a field by one Gear/BDF2 time step.

        Args:
            field: Field to update in place.
            dtloc: Scalar or local time-step array.
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

List_LSRK_Integrators = [lsrk25bb, lsrk26bb]
List_RK_Integrators = [rk2, rk2_heun, rk3_heun, rk3ssp, rk4] + List_LSRK_Integrators
List_Explicit_Integrators = [explicit] + List_RK_Integrators
List_Implicit_Integrators = [implicit, cranknicolson, gear]
