import numpy as np
from typing import Union

'''
matchers recieve world sensor output, and belief sensor output and compare them
'''

def lidar1D_matcher(world_z : np.ndarray, world_solid_names : list[str],
          belief_z : np.ndarray, belief_solid_names : list[str]) -> Union[list[bool],list[bool]]:

    exist_solid_names = []
    notexist_solid_names = []
    
    for wsn,bsn in zip(world_solid_names, belief_solid_names):
        if wsn != bsn and bsn:
            notexist_solid_names.append(bsn)
        if wsn:
            exist_solid_names.append(wsn)

    return exist_solid_names, notexist_solid_names