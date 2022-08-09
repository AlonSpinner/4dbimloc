import numpy as np
from typing import Union
from bim4loc.solids import IfcSolid
from . import probalistic_models as pm

'''
matchers recieve world sensor output, and belief sensor output and compare them
'''

def lidar1D_matcher(world_z : np.ndarray,
          belief_z : np.ndarray, belief_solid_names : list[str], 
          sensor_std,
          world_solid_names  = []) -> Union[list[bool],list[bool]]:

    exist_solid_names = []
    notexist_solid_names = []
    
    #-------------------- Ground Truth-------------------------
    # for wsn,bsn in zip(world_solid_names, belief_solid_names):
    #     if wsn != bsn and bsn:
    #         notexist_solid_names.append(bsn)
    #     if wsn:
    #         exist_solid_names.append(wsn)

    
    for wz, bz, bsn in zip(world_z, belief_z, belief_solid_names):
        # p = pm.m_given_rangeWorld_rangeBelief("⬛", wz, bz, 3*sensor_std)
        if abs(bz - wz) < 3 * sensor_std:
            exist_solid_names.append(bsn)
        else:
            notexist_solid_names.append(bsn)

    return exist_solid_names, notexist_solid_names
