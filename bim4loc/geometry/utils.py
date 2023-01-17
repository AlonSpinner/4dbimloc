#based on: https://stackoverflow.com/questions/58666635/implementing-pca-with-numpy
import numpy as np
from numba import njit, prange

@njit(cache = True)
def pca(X, n_components=2):
    """
    Assumes observations in X are passed as rows of a numpy array.
    """

    # Translate the dataset so it's centered around 0
    translated_X = X - np.mean(X)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    e_values, e_vectors = np.linalg.eigh(np.cov(translated_X.T))

    # Sort eigenvalues and their eigenvectors in descending order
    e_ind_order = np.flip(e_values.argsort())
    # e_values = e_values[e_ind_order]
    e_vectors = e_vectors[e_ind_order]

    # Save the first n_components eigenvectors as principal components
    principal_components = e_vectors[:n_components,:] #np.take(e_vectors, np.arange(n_components), axis=0)

    transform = np.ascontiguousarray(principal_components.T)
    X_hat = translated_X @ transform
    return X_hat, transform

@njit(cache = True)
def point_in_polygon(point : np.ndarray, polygon : np.ndarray):
    """
    Determine if a point is inside a given polygon or not
    Polygon is a list of (x,y) pairs.

    there is also this: https://www.linkedin.com/pulse/short-formula-check-given-point-lies-inside-outside-polygon-ziemecki/
    """
    n = len(polygon)
    inside =False

    p1x,p1y = polygon[0]
    for i in prange(n+1):
        p2x,p2y = polygon[i % n]
        if point[1] > min(p1y,p2y):
            if point[1] <= max(p1y,p2y):
                if point[0] <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (point[1]-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or point[0] <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside