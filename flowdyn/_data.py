# -*- coding: utf-8 -*-
"""
    The ``_data`` module 
    =========================

    Provides scalar and vector computations

    :Example:

    Available functions
    -------------------

 """

import numpy as np

__all__ = ['_vecmag', '_vecsqrmag', '_sca_mult_vec', '_vec_dot_vec', 'datavector']

# ===============================================================
def _vecmag(qdata):
    return np.sqrt(np.sum(qdata**2, axis=0))

def _vecsqrmag(qdata):
    return np.sum(qdata**2, axis=0)

def _sca_mult_vec(r, v):
    return r*v # direct multiplication thanks to shape (:)*(2,:)

def _vec_dot_vec(v1, v2):
    return np.einsum('ij,ij->j', v1, v2)

def datavector(ux, uy, uz=None):
    return np.vstack([ux, uy]) if not uz else np.vstack([ux, uy, uz])

