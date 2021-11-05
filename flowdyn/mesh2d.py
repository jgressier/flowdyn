# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:42:29 2013

@author: j.gressier
"""

import numpy as np
import flowdyn.meshbase as meshbase

class mesh2d(meshbase.virtualmesh):
    """
    cartesian uniform 2D mesh
    cells are sorted row-wise (i is fast index), global index is: nx*j+i
    faces are ordered as i/x varying vertical faces ny*(nx+1) followed by j/y (ny+1)*nx faces
        index of related cell has the same order (i fast index, as rows)
    """
    def __init__(self, nx, ny, lx=1., ly=1.):
        meshbase.virtualmesh.__init__(self, type='2D')
        self.nx    = nx
        self.ny    = ny
        self.ncell = nx*ny
        self.lx    = lx
        self.ly    = ly
        self._bctags = ['top', 'bottom', 'left', 'right' ]
        self._bcfaces_orientation = {'top':'outward', 'bottom':'inward', 'left':'inward', 'right':'outward' }
        self._io_bcfaces = {
            'left'   :  np.arange(ny)*(nx+1),
            'right'  : (np.arange(ny)+1)*(nx+1)-1,
            'top'    : ny*(nx+1) + ny*nx + np.arange(nx),
            'bottom' : ny*(nx+1) + np.arange(nx) }

    def nbfaces(self):
        "returns number of faces"
        return (self.nx+1)*self.ny + self.nx*(self.ny+1)

    def centers(self):
        "compute centers of cells in a mesh"
        x = np.linspace(0., self.lx, self.nx, endpoint=False)+ .5*self.dx()
        y = np.linspace(0., self.ly, self.ny, endpoint=False)+ .5*self.dy()
        xx, yy = np.meshgrid(x, y)
        return xx.flatten(), yy.flatten()

    def dx(self):
        "compute cell sizes in a mesh"
        return self.lx / self.nx

    def dy(self):
        "compute cell sizes in a mesh"
        return self.ly / self.ny

    def vol(self):
        "compute cell sizes in a mesh"
        # dx = np.zeros(self.ncell)
        # for i in np.arange(self.ncell):
        #     dx[i] = (self.xf[i+1]-self.xf[i])
        return np.repeat(self.dx()*self.dy(), self.ncell)

    def __repr__(self):
        print("mesh object: mesh2d")
        # print("length : ", self.length)
        # print("ncell  : ", self.ncell)
        # dx = self.dx()
        # print("min dx : ", dx.min())
        # print("max dx : ", dx.max())
        # print("min dy : ", dy.min())
        # print("max dy : ", dy.max())        

    def list_of_bctags(self):
        """
        return list of BC tags
        """
        return self._bctags

    def index_of_bc(self, tag):
        """
        return list of index of faces for a given boundary condition
        """
        return self._io_bcfaces[tag]

    def bcface_orientation(self, tag):
        """
        return list of index of faces for a given boundary condition
        """
        return self._bcfaces_orientation[tag]

    def normal_of_bc(self, tag):
        """
        return normal array for all faces of tag BC
        """
        if tag in ['top','bottom']:
            dir = np.zeros((2,self.nx))
            dir[1,:] = 1.
        elif tag in ['left','right']:
            dir = np.zeros((2,self.ny))
            dir[0,:] = 1.
        else:
            raise NameError("unexpected tag "+tag)
        if self._bcfaces_orientation[tag] == 'inward':
            dir = -dir
        return dir

class unimesh(mesh2d): # alias
    pass