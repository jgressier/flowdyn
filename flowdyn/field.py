# -*- coding: utf-8 -*-
"""module field

"""
__all__ = ["fdata"]

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("unable to import matplotlib, some features will be missing")

# import model
# import mesh

class fdata:
    """define field: neq x nelem data
      model : number of equations
      mesh  : mesh
      data  : data to initialize

    Args:

    Returns:

    """

    def __init__(self, model, mesh, data=None, t=0.0, it=-1):
        self.model = model
        self.neq = model.neq
        self.mesh = mesh
        self.nelem = mesh.ncell
        self.time = t
        self.it   = it
        if data is not None:
            self.data = data[:]  # copy shape
            # and check
            for i, d in enumerate(data):
                if np.ndim(d) < self.model.shape[i]:
                    self.data[i] = np.repeat(
                        np.expand_dims(d, axis=0), self.nelem, axis=0
                    ).T
                else:
                    self.data[i] = d.copy()
            # self.data = [ np.array(d).T*np.ones(self.nelem) for d in data ] # old version only working for scalars
        else:
            raise NotImplementedError("no more possible to get data signature")
            # self.data = [ np.zeros(self.nelem) ] * self.neq
            # for i in range(self.neq):
            #     self.data.append(np.zeros(nelem))

    def copy(self):
        """ returns copy of current instance """
        new = fdata(self.model, self.mesh, self.data, 
                t=self.time, it=self.it)
        return new

    def set(self, f):
        """set (as a reference) all members of a fielf to current field

        Args:
          f: field

        Returns:

        """
        self.__init__(f.model, f.mesh, f.data,
                     t=f.time, it=f.it)

    def set_time(self, time):
        self.time = time

    def reset(self, t=0., it=-1):
        self.time=t
        self.it=it

    def interpol_t(self, f, t):
        """create a new field time-interpolated between self and f

        Args:
            f (field): field to interpolate to
            t (float): time to interpolate
        Returns:
            new interpolated field
        """
        new = self.copy()
        new.it = -1 # don't know how to define
        k = (t-self.time)/(f.time-self.time)
        new.time = t
        for i in range(f.neq):
            new.data[i] += k * (f.data[i]-self.data[i])
        return new

    def diff(self, f):
        """create a new field time-interpolated between self and f

        Args:
            f (field): field to interpolate to
            t (float): time to interpolate
        Returns:
            new interpolated field
        """
        new = self.copy()
        new.it = -1
        new.time -= f.time
        for i in range(f.neq):
            new.data[i] -= f.data[i]
        return new

    def zero_datalist(self, newdim=None):
        """returns a list of numpy.array with the same shape of self.data, possibly resizes to dim if provided

        Args:
          newdim:  (Default value = None)

        Returns:

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
        """check nan valies is all solution field"""
        return any([np.any(np.isnan(d)) for d in self.data])

    def phydata(self, name):
        """returns the numpy array of given physical name, according to self.model

        Args:
          name: name of physical data, available in model.list_var()

        Returns:

        """
        return self.model.nameddata(name, self.data)

    def plot(self, name, style="o", axes=plt):
        """plot named physical date along x axis of internal mesh

        Args:
          name: name of physical data, available in model.list_var()
          style:  (Default value = 'o')
          axes: specify optional axes system (Default value = plt)

        Returns:

        """
        return axes.plot(self.mesh.centers(), self.phydata(name), style)
    
    def plot2dcart(self, name, style='o', axes=plt): #basic idea on how to get a plot based on 2D FVM while using a 1D case.
        xx,yy = self.mesh.centers()
        return axes.plot(xx[0:self.mesh.nx], self.phydata(name)[0:self.mesh.nx], style)    

    def semilogy(self, name, style="o", axes=plt):
        """plot named physical date along x axis of internal mesh

        Args:
          name: name of physical data, available in model.list_var()
          style:  (Default value = 'o')
          axes: specify optional axes system (Default value = plt)

        Returns:

        """
        return axes.semilogy(self.mesh.centers(), self.phydata(name), style)

    def average(self, name):
        """Computes average named data

        Args:
          name: name of physical data, available in model.list_var()

        Returns: average (cell volume weighted)

        """
        return self.mesh.average(self.phydata(name))
           
    def stats(self, name):
        """Computes average and variance of named data

        Args:
          name: name of physical data, available in model.list_var()

        Returns:

        """
        avg = self.mesh.average(self.phydata(name))
        var = self.mesh.average((self.phydata(name) - avg) ** 2)
        return avg, var
        
    def contour(self, name, style={}, axes=None):
        """draw contour lines from 2d data

        Args:
          name:
          style:  (Default value = {})
          axes:  (Default value = plt)

        Returns:

        """
        if axes is None: axes=plt.gca()
        xx, yy = self.mesh.centers()
        axes.set_aspect('equal')
        return axes.contour(
            xx.reshape((self.mesh.ny, self.mesh.nx)),
            yy.reshape((self.mesh.ny, self.mesh.nx)), 
            self.phydata(name).reshape((self.mesh.ny, self.mesh.nx)))

    def contourf(self, name, style={}, axes=None):
        """draw flooded contour from 2d data

        Args:
          name:
          style:  (Default value = {})
          axes:  (Default value = plt)

        Returns:

        """
        if axes is None: axes=plt.gca()
        # TODO must check this is a 2D mesh
        xx, yy = self.mesh.centers()
        axes.set_aspect("equal")
        return axes.contourf(
            xx.reshape((self.mesh.ny, self.mesh.nx)),
            yy.reshape((self.mesh.ny, self.mesh.nx)), 
            self.phydata(name).reshape((self.mesh.ny, self.mesh.nx)),
        )

    def set_plotdata(self, line, name):
        """apply data to line object (often for animations)

        Args:
          line:
          name:

        Returns:

        """
        line.set_data(self.mesh.centers(), self.phydata(name))
        return

class fieldlist():
    """define field list: result of solver integration
        can be handled as a list object but add some specific functions
    Args:

    Returns:

    """

    def __init__(self):
        self.solutions = list()
        self._packed = False # not yet used

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

    def xtcontour(self, varname, levels=20, axes=None, style={}):
        xc = self.solutions[0].mesh.centers()
        tt = self.time_array()
        xx, xt = np.meshgrid(xc, tt)
        solgrid = [ s.phydata(varname) for s in self.solutions ]
        if axes is None: axes=plt.gca()
        axes.contour(xx, xt, solgrid, levels=levels, **style)

    def xtcontourf(self, varname, levels=20, axes=None, style={}):
        xc = self.solutions[0].mesh.centers()
        tt = self.time_array()
        xx, xt = np.meshgrid(xc, tt)
        solgrid = [ s.phydata(varname) for s in self.solutions ]
        if axes is None: axes=plt.gca()
        axes.contourf(xx, xt, solgrid, levels=levels, **style)
