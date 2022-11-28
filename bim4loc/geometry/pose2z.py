import numpy as np
from numba import njit, prange

# s - [x,y,z,theta]

@njit(cache = True)
def T_from_Rt(R : np.ndarray, t : np.ndarray) -> np.ndarray:
    T = np.array([[R[0,0], R[0,1], R[0,2], t[0]],
                  [R[1,0], R[1,1], R[1,2], t[1]],
                  [R[2,0], R[2,1], R[2,2], t[2]],
                  [0.0, 0.0, 0.0, 1.0]])
    return T

@njit(cache = True)
def R_from_theta(theta : np.ndarray) -> np.ndarray:
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    return R

@njit(cache = True)
def T_from_s(s):
    R = R_from_theta(s[3])
    t = s[:3]
    T = T_from_Rt(R, t)
    return T

@njit(cache = True)
def s_from_T(T : np.ndarray) -> np.ndarray:
    s = np.array((T[0,3],T[1,3], T[2,3], np.arctan2(T[1,0], T[0,0])))
    return s

@njit(cache = True)
def s_from_Rt(R : np.ndarray, t : np.ndarray) -> np.ndarray:
    s = np.array((t[0,0],t[1,0], t[2,0], np.arctan2(R[1,0], R[0,0])))
    return s

@njit(cache = True)
def compose_s(s : np.ndarray, ds : np.ndarray)  -> np.ndarray:
    Ts = T_from_s(ds)
    Tds = T_from_s(s)
    T = Tds @ Ts
    s = s_from_T(T)
    return s

@njit(cache = True)
def compose_s_array(s_array : np.ndarray, ds : np.ndarray)  -> np.ndarray:
    #s_array is of shape (N,4)
    Ts = T_from_s(ds)
    new_s_array = np.zeros_like(s_array)
    for i in prange(s_array.shape[0]):
        Tds = T_from_s(s_array[i])
        T = Tds @ Ts
        new_s_array[i] = s_from_T(T)
    return new_s_array

@njit(cache = True)
def transform_from(s : np.ndarray, p : np.ndarray)  -> np.ndarray:
    # p - 3 x m
    R = R_from_theta(s[3])
    t = s[:3].reshape(3,1)
    return R @ p + t

