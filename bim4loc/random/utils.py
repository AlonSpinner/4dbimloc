import numpy as np
from numba import njit, prange

EPS = 1e-16

@njit(cache = True)
def negate(p):
    return 1.0 - p

@njit(cache = True)
def odds2p(odds):
    return odds / (1.0 + odds)

@njit(cache = True)
def p2odds(p):
    return p /np.maximum(1.0 - p, EPS)

@njit(cache = True)
def p2logodds(p):
    return np.log(p2odds(p))

@njit(cache = True)
def logodds2p(l):
    l = np.minimum(l, 5.0)
    return  np.exp(l) / (1.0 + np.exp(l))

# @njit(cache = True)
def compute_entropy(p : np.ndarray) -> np.ndarray:
    if p.ndim == 1:
        entropy = 0.0
        for i in prange(p.shape[0]):
            p[i] = np.clip(p[i], 0.0, 1.0)
            if p[i] != 0.0:
                entropy = entropy - p[i] * np.log2(p[i])
            if p[i] != 1.0:
                entropy = entropy - (1.0 - p[i]) * np.log2(1.0 - p[i])
            if entropy == np.NaN:
                a = 1
        return np.array([entropy])
    elif p.ndim == 2:
        entropy = np.zeros(p.shape[0])
        for i in prange(p.shape[0]):
            for j in prange(p.shape[1]):
                p[i,j] = np.clip(p[i,j], 0.0, 1.0)
                if p[i,j] != 0.0:
                    entropy[i] = entropy[i] - p[i,j] * np.log2(p[i,j])
                if p[i,j] != 1.0:
                    entropy[i] = entropy[i] - (1.0 - p[i,j]) * np.log2(1.0 - p[i,j])
    return entropy

def compute_cross_entropy(p : np.ndarray, q : np.ndarray) -> np.ndarray:
    if p.ndim == 1:
        cross_entropy = 0.0
        for i in prange(p.shape[0]):
            q[i] = np.clip(q[i], 0.01, 0.99)
            if p[i] != 0.0:
                cross_entropy = cross_entropy - p[i] * np.log2(q[i])
            if p[i] != 1.0:
                cross_entropy = cross_entropy - (1.0 - p[i]) * np.log2(1.0 - q[i])
        return np.array([cross_entropy])
    elif p.ndim == 2:
        cross_entropy = np.zeros(p.shape[0])
        for i in prange(p.shape[0]):
            for j in prange(p.shape[1]):
                q[i,j] = np.clip(q[i,j], 0.01, 0.99)
                if p[i,j] != 0.0:
                    cross_entropy[i] = cross_entropy[i] - p[i,j] * np.log2(q[i,j])
                if p[i,j] != 1.0:
                    cross_entropy[i] = cross_entropy[i] - (1.0 - p[i,j]) * np.log2(1.0 - q[i,j])
    return cross_entropy

def hamming_distance(v1 : np.ndarray,v2 : np.ndarray):
    '''
    v1 and v2 are binary vectors
    '''
    return np.sum(np.abs(v1 - v2))

def jaccard_distance(x, y):
    x_set = set([i for i in range(len(x)) if x[i]])
    y_set = set([i for i in range(len(y)) if y[i]])
    return 1 - (len(x_set.intersection(y_set)) / len(x_set.union(y_set)))
