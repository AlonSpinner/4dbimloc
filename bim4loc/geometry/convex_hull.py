import numpy as np
from numba import njit, prange

#-----------------------------------------------OPTION 1-----------------------------------------------
# from scipy.spatial import ConvexHull
# def convex_hull(points : np.ndarray) -> np.ndarray:
#     '''
#     accepts and returns a numpy array of shape (N, 2)
#     '''
#     hull = ConvexHull(points)
#     return points[hull.vertices]

#-----------------------------------------------OPTION 2-----------------------------------------------
#from https://stackoverflow.com/questions/74812556/computing-quick-convex-hull-using-numba

#-----------------------------------------------OPTION 3-----------------------------------------------
@njit(cache = True)
def convex_hull_jarvis(points):
    #jarvis march, coded by chatgpt
    n = len(points)
    # Initialize the starting point of the convex hull to be the leftmost point
    start = 0
    for i in range(1, n):
        if points[i][0] < points[start][0]:
            start = i
    hull = []
    current = start
    while True:
        hull.append(current)
        next_point = (current + 1) % n
        for i in prange(n):
            if (points[next_point][1] - points[hull[-1]][1]) * (points[i][0] - points[hull[-1]][0]) > (points[next_point][0] - points[hull[-1]][0]) * (points[i][1] - points[hull[-1]][1]):
                next_point = i
        current = next_point
        if current == start:
            break
    return points[np.array(hull)]
#-----------------------------------------------OPTION 4-----------------------------------------------
@njit(cache = True)
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

@njit(cache = True)
def sort_points(points):
    """
    sort the points by polar angle
    """
    n = points.shape[0]
    ymin = points[np.argmin(points[:, 1]), 1]
    angle = np.zeros(n)
    for i in prange(n):
        angle[i] = np.arctan2(points[i, 1] - ymin, points[i, 0])
    indices = np.argsort(angle)
    return indices

@njit(cache = True)
def convex_hull_chan(points):
    
    indices = sort_points(points)
    n = points.shape[0]
    k = 0
    hull = np.empty(n + 1, np.int32)
    # lower convex hull
    for i in prange(n):
        while k >= 2 and cross(points[hull[k - 2]], points[hull[k - 1]], points[indices[i]]) < 0:
            k -= 1
        hull[k] = indices[i]
        k += 1
    # upper convex hull
    for i in prange(n - 2, -1, -1):
        while k >= 2 and cross(points[hull[k - 2]], points[hull[k - 1]], points[indices[i]]) <= 0:
            k -= 1
        hull[k] = indices[i]
        k += 1
    return points[hull[:k - 1]]