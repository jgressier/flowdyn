# -*- coding: utf-8 -*-
"""Provide field containers for spatial and time-dependent solution data."""

__all__ = ["fdata"]

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("unable to import matplotlib, some features will be missing")

# import model
# import mesh


class fdata:
    """Store the equation data associated with a mesh and physical model.

    Args:
        model: Physical model defining the equations and component shapes.
        mesh: Mesh on which the data are defined.
        data: Initial conservative data, one entry per equation.
        t: Physical time associated with the field.
        it: Iteration number associated with the field.
    """

    def __init__(self, model, mesh, data=None, t=0.0, it=-1):
        self.model = model
        self.neq = model.neq
        self.mesh = mesh
        self.nelem = mesh.ncell
        self.time = t
        self.it = it
        if data is not None:
            if len(data) != self.neq:
                raise ValueError(f"expected {self.neq} data components, got {len(data)}")
            self.data = data[:]  # copy shape
            # and check
            for i, d in enumerate(data):
                if np.ndim(d) < self.model.shape[i]:
                    self.data[i] = np.repeat(np.expand_dims(d, axis=0), self.nelem, axis=0).T
                else:
                    self.data[i] = d.copy()
                if self.data[i].shape[-1] == 1 and self.nelem != 1:
                    self.data[i] = np.repeat(self.data[i], self.nelem, axis=-1)
                if self.data[i].shape[-1] != self.nelem:
                    raise ValueError(
                        f"component {i} has {self.data[i].shape[-1]} cells; " f"mesh has {self.nelem}"
                    )
            # self.data = [ np.array(d).T*np.ones(self.nelem) for d in data ] # old version only working for scalars
        else:
            raise NotImplementedError("no more possible to get data signature")
            # self.data = [ np.zeros(self.nelem) ] * self.neq
            # for i in range(self.neq):
            #     self.data.append(np.zeros(nelem))

    def copy(self):
        """Return an independent copy of the field data."""
        new = fdata(self.model, self.mesh, self.data, t=self.time, it=self.it)
        return new

    def set(self, f):
        """Replace all field members with values from another field.

        Args:
            f: Field whose values should be copied.
        """
        self.__init__(f.model, f.mesh, f.data, t=f.time, it=f.it)

    def set_time(self, time):
        self.time = time

    def reset(self, t=0.0, it=-1):
        self.time = t
        self.it = it

    def interpol_t(self, f, t):
        """Interpolate a new field between this field and another one.

        Args:
            f: Field defining the other interpolation endpoint.
            t: Time at which to interpolate.

        Returns:
            The interpolated field.
        """
        new = self.copy()
        new.it = -1  # don't know how to define
        k = (t - self.time) / (f.time - self.time)
        new.time = t
        for i in range(f.neq):
            new.data[i] += k * (f.data[i] - self.data[i])
        return new

    def diff(self, f):
        """Compute the difference between this field and another one.

        Args:
            f: Field to subtract.

        Returns:
            A new field containing the data and time differences.
        """
        new = self.copy()
        new.it = -1
        new.time -= f.time
        for i in range(f.neq):
            new.data[i] -= f.data[i]
        return new

    def zero_datalist(self, newdim=None):
        """Create zero arrays matching the shapes of the field components.

        Args:
            newdim: Optional replacement for the last dimension.

        Returns:
            A list of zero-filled NumPy arrays.
        """
        if newdim:
            datalist = [0 for d in self.data]
            for i, d in enumerate(self.data):
                newshape = np.array(d.shape)
                newshape[-1] = newdim
                datalist[i] = np.zeros(newshape)
        else:
            datalist = [np.zeros(d.shape) for d in self.data]
        return datalist

    def isnan(self):
        """Return whether any solution component contains a NaN value."""
        return any([np.any(np.isnan(d)) for d in self.data])

    def phydata(self, name):
        """Return the physical variable identified by a model-defined name.

        Args:
            name: Variable name exposed by ``model.list_var()``.

        Returns:
            The requested physical data array.
        """
        return self.model.nameddata(name, self.data)

    def plot(self, name, style="o", axes=plt):
        """Plot a physical variable along the mesh x-axis.

        Args:
            name: Variable name exposed by ``model.list_var()``.
            style: Matplotlib line style.
            axes: Matplotlib plotting object or axes.

        Returns:
            Matplotlib line objects created by the plot operation.
        """
        return axes.plot(self.mesh.centers(), self.phydata(name), style)

    def plot2dcart(
        self, name, style='o', axes=plt
    ):  # basic idea on how to get a plot based on 2D FVM while using a 1D case.
        xx, yy = self.mesh.centers()
        return axes.plot(xx[0 : self.mesh.nx], self.phydata(name)[0 : self.mesh.nx], style)

    def semilogy(self, name, style="o", axes=plt):
        """Plot a physical variable with a logarithmic y-axis.

        Args:
            name: Variable name exposed by ``model.list_var()``.
            style: Matplotlib line style.
            axes: Matplotlib plotting object or axes.

        Returns:
            Matplotlib line objects created by the plot operation.
        """
        return axes.semilogy(self.mesh.centers(), self.phydata(name), style)

    def average(self, name):
        """Compute the cell-volume-weighted average of a physical variable.

        Args:
            name: Variable name exposed by ``model.list_var()``.

        Returns:
            The cell-volume-weighted average.
        """
        return self.mesh.average(self.phydata(name))

    def stats(self, name):
        """Compute the average and variance of a physical variable.

        Args:
            name: Variable name exposed by ``model.list_var()``.

        Returns:
            A tuple containing the cell-volume-weighted average and variance.
        """
        avg = self.mesh.average(self.phydata(name))
        var = self.mesh.average((self.phydata(name) - avg) ** 2)
        return avg, var

    def contour(self, name, style=None, axes=None):
        """Draw contour lines for two-dimensional physical data.

        Args:
            name: Variable name exposed by ``model.list_var()``.
            style: Reserved for plot styling compatibility.
            axes: Matplotlib axes. The current axes are used when omitted.

        Returns:
            The generated Matplotlib contour set.
        """
        if axes is None:
            axes = plt.gca()
        xx, yy = self.mesh.centers()
        axes.set_aspect('equal')
        return axes.contour(
            xx.reshape((self.mesh.ny, self.mesh.nx)),
            yy.reshape((self.mesh.ny, self.mesh.nx)),
            self.phydata(name).reshape((self.mesh.ny, self.mesh.nx)),
        )

    def contourf(self, name, style=None, axes=None):
        """Draw filled contours for two-dimensional physical data.

        Args:
            name: Variable name exposed by ``model.list_var()``.
            style: Reserved for plot styling compatibility.
            axes: Matplotlib axes. The current axes are used when omitted.

        Returns:
            The generated Matplotlib contour set.
        """
        if axes is None:
            axes = plt.gca()
        # TODO must check this is a 2D mesh
        xx, yy = self.mesh.centers()
        axes.set_aspect("equal")
        return axes.contourf(
            xx.reshape((self.mesh.ny, self.mesh.nx)),
            yy.reshape((self.mesh.ny, self.mesh.nx)),
            self.phydata(name).reshape((self.mesh.ny, self.mesh.nx)),
        )

    def set_plotdata(self, line, name):
        """Apply current field data to an existing Matplotlib line.

        Args:
            line: Matplotlib line object to update.
            name: Variable name exposed by ``model.list_var()``.
        """
        line.set_data(self.mesh.centers(), self.phydata(name))
        return


class fieldlist:
    """Store an ordered collection of fields produced by time integration."""

    statsfuncs = {'min': np.min, 'max': np.max}

    def __init__(self):
        self.solutions = list()
        self._packed = False  # not yet used
        self._stats = {}

    def __getitem__(self, i):
        return self.solutions[i]

    def __len__(self):
        return len(self.solutions)

    def append(self, s):
        self._packed = False
        self.solutions.append(s)

    def extend(self, flist):
        self._packed = False
        self.solutions.extend(flist.solutions)

    def time_array(self):
        return [s.time for s in self.solutions]

    def it_array(self):
        return [s.it for s in self.solutions]

    def stack_solution(self, varname):
        return [s.phydata(varname) for s in self.solutions]

    def stats_solutions(self, varname):
        self._stats[varname] = {}
        sols = self.stack_solution(varname)
        for key, func in self.statsfuncs.items():
            self._stats[varname][key] = func(sols)
        return self._stats[varname]

    def xtcontour(self, varname, levels=20, axes=None, style=None):
        xc = self.solutions[0].mesh.centers()
        tt = self.time_array()
        xx, xt = np.meshgrid(xc, tt)
        solgrid = self.stack_solution(varname)
        if axes is None:
            axes = plt.gca()
        style = {} if style is None else style
        axes.contour(xx, xt, solgrid, levels=levels, **style)

    def xtcontourf(self, varname, levels=20, axes=None, style=None):
        xc = self.solutions[0].mesh.centers()
        tt = self.time_array()
        xx, xt = np.meshgrid(xc, tt)
        solgrid = [s.phydata(varname) for s in self.solutions]
        if axes is None:
            axes = plt.gca()
        style = {} if style is None else style
        axes.contourf(xx, xt, solgrid, levels=levels, **style)
