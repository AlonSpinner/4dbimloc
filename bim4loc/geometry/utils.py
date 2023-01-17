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
    for i in prange(e_vectors.shape[1]):
        e_vectors[:,i] = e_vectors[:,i] / np.linalg.norm(e_vectors[:,i])

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
def T_from_pitch_yaw(pitch, yaw):
    '''
    pitch - rotation around x axis
    yaw - rotation around y axis
    '''
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw), np.cos(yaw), 0],
                  [0, 0, 1]])
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    T = np.zeros((4,4)); T[3,3] = 1.0
    T[:3,:3] = R_yaw @ R_pitch
    return T

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

@njit(cache = True)
def distance_to_line(p0,p1,q):
    #p0 and p1 are two points on the line
    #q is the point we want to find the distance to
    #https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    projected_point = p0 + np.dot(q-p0,p1-p0)/np.dot(p1-p0,p1-p0)*(p1-p0)
    distance = np.linalg.norm(projected_point-q)
    return distance, projected_point