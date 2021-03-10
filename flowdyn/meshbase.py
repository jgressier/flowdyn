# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:42:29 2013

@author: j.gressier
"""

import sys
import math
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

    def __repr__(self):
        raise NotImplementedError()

