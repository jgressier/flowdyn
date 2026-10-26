# -*- coding: utf-8 -*-
"""Provide scalar and vector operations for multicomponent field arrays."""

import numpy as np

__all__ = ['_vecmag', '_vecsqrmag', '_sca_mult_vec', '_vec_dot_vec', 'datavector']


# ===============================================================
def _vecmag(qdata):
    return np.sqrt(np.sum(qdata**2, axis=0))


def _vecsqrmag(qdata):
    return np.sum(qdata**2, axis=0)


def _sca_mult_vec(r: float, v: np.ndarray):
    return r * v  # direct multiplication thanks to shape (:)*(2,:)


def _vec_dot_vec(v1: np.ndarray, v2: np.ndarray):
    return np.einsum('ij,ij->j', v1, v2)


def datavector(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray = None) -> np.ndarray:
    return np.vstack([ux, uy]) if uz is None else np.vstack([ux, uy, uz])
