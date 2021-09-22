# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:42:29 2013

@author: j.gressier
"""

#import sys
#import math
import numpy as np

class virtualmesh():
    """
    virtual class for a domain and its mesh
    """

    def __init__(self, type='virtual'):
        self._type = type

    def nbfaces(self):
        "returns number of faces"
        raise NotImplementedError()

    def centers(self):
        "compute centers of cells in a mesh"
        raise NotImplementedError()

    def vol(self):
        "compute cell sizes in a mesh"
        raise NotImplementedError()

    def average(self, data):
        "volume weighted average of data"
        return np.average(data, weights=self.vol())

    def L1average(self, data):
        "volume weighted average of data"
        return np.average(np.abs(data), weights=self.vol())

    def L2average(self, data):
        "volume weighted average of data"
        return np.sqrt(np.average(data**2, weights=self.vol()))

    def __repr__(self):
        raise NotImplementedError()
