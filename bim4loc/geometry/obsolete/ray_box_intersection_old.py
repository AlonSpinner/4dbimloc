import numpy as np
from numba import prange
def ray_box_intersection(ray_o : np.ndarray, ray_inv_dir : np.ndarray, box : np.ndarray) -> bool:
    '''
    based on https://tavianator.com/2022/ray_box_boundary.html

    input:
        ray_o - np.array([x,y,z])
        ray_inv_dir - np.array([dx,dy,dz]), computed as 1/ray_dir and may include infs.
        box - np.array([minx,miny,minz,maxx,maxy,maxz])

    output:
        boolean - True if ray intersects box          
    '''
    tmin = -np.inf; tmax = np.inf

    for i in prange(3):
        t1 = (box[i]- ray_o[i]) * ray_inv_dir[i]
        t2 = (box[i+3] - ray_o[i]) * ray_inv_dir[i]
    
        #imortant: max(1,np.inf) -> 1 
        #          max(1,np.nan) -> nan
        # therfore tmin/tmax are always the first value in the comparison
        tmin = min(max(tmin, t1), max(tmin, t2))
        tmax = max(min(tmax, t1), min(tmax, t2))
    
    return abs(tmin) <= abs(tmax)