# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:36:35 2013

@author: j.gressier
"""
import numpy as np

class model():
    def __init__(self, equation='convection'):
        self.equation = equation  # convection, diffusion or burgers
        self.convcoef = 1.
        self.diffcoef = 0.
        self.islinear = 0

    def __repr__(self):
        print "model: ", self.equation
        print "convection coefficient: ", self.convcoef
        print "diffusion  coefficient: ", self.diffcoef
        
    def cons2prim(self):
        print "cons2prim method not implemented"
    
    def prim2cons(self):
        print "prim2cons method not implemented"
    
    def numflux(self):
        pass
    
    def timestep(self, pdata, dx, condition):
        pass

class convmodel(model):
    def __init__(self, convcoef):
        self.equation = 'convection'
        self.neq      = 1
        self.convcoef = convcoef
        self.islinear = 1
        
    def cons2prim(self, qdata):
        return qdata
        
    def prim2cons(self, pdata):
        return pdata

    def numflux(self, pL, pR):
        return [ self.convcoef*(pL[0]+pR[0])/2-abs(self.convcoef)*(pR[0]-pL[0])/2 ]
    
    def timestep(self, data, dx, condition):
        "computation of timestep: data is not used, dx is an array of cell sizes, condition is the CFL number"
        return condition*dx/abs(self.convcoef)
        
        