# -*- coding: utf-8 -*-
"""
xnum: package for spatial numerical methods
"""

import numpy as np
import mesh

class virtualmeth():
    def __init__(self):
        pass
    
    def interp_face(self, mesh, data):
        pass
    
class extrapol(virtualmeth):
    "first order method"
    def interp_face(self, mesh, data):
        "returns 2x (L/R) neq list of (ncell+1) nparray"
        nc = data[0].size
        if (mesh.ncell <> nc): print self.__class__+"/interp_face error: mismatch sizes"
        Ldata = []
        Rdata = []
        for i in range(len(data)):
            Ldata.append(np.zeros(nc+1))
            Rdata.append(np.zeros(nc+1))
            Ldata[i][1:nc+1] = data[i][:]
            Rdata[i][0:nc]   = data[i][:]
            #print 'L/R',Ldata[i].size, Rdata[i].size
        return Ldata, Rdata