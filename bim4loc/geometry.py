import numpy as np

class pose2z:

    size = 4 #state size

    def __init__(self, x, y, theta, z) -> None:
        #we hold the storage form
        self._x : float = x
        self._y : float = y
        self._theta : float = theta
        self._z : float = z
        
        self.R = self.Rz(theta)
        self.t = np.vstack((x, y, z))

    @classmethod
    def from_Exp(cls, T : np.ndarray) -> 'pose2z':
        #T is a 4x4 transformation matrix
        x = T[0,3]; y = T[1,3]; z = T[2,3]
        theta = np.arctan2(T[1,0], T[0,0])
        return cls(x ,y ,theta, z)

    @classmethod
    def from_Rt(cls, R : np.ndarray, t : np.ndarray) -> 'pose2z':
        h = np.array([0,0,0,1])
        return cls.from_Exp(cls.Rt2T(R,t))

    @staticmethod
    def identity() -> 'pose2z':
        return pose2z(0,0,0,0)

    def Exp(self) -> np.ndarray:
        return self.Rt2T(self.R,self.t)

    def Log(self) -> np.ndarray:
        #slow but correct as opposed to just returnning internal storage.. whatevah
        theta = np.arctan2(self.R[1,0], self.R[0,0])
        return np.array([np.asscalar(self.t[0]), np.asscalar(self.t[1]), theta, np.asscalar(self.t[2])])

    def compose(self, g : 'pose2z') -> 'pose2z':
        #returns the composition of self and x.
        return self.from_Exp(self.Exp() @ g.Exp())

    def inverse(self) -> 'pose2z':
        #returns the inverse of self.
        invR = self.R.T
        invt = -invR @ self.t
        return self.from_Rt(invR,invt)

    def between(self, g : 'pose2z') -> 'pose2z':
        return self.compose(self.inverse(),g)

    def transform_to(self, p : np.ndarray) -> np.ndarray:
        return self.Exp() @ p

    def transform_from(self, p : np.ndarray) -> np.ndarray:
        return self.inverse().Exp() @ p

    def localCoordinates(self, x : 'pose2z') -> 'pose2z':
        return self.between(x).Log()

    def retract(self, v : np.ndarray) -> 'pose2z':
        return self.compose(pose2z(*v))

    def __mul__(self, x : 'pose2z') -> 'pose2z':
        return self.compose(x)
    
    def __add__(self, v : np.ndarray) -> 'pose2z':
        return self.retract(v)

    def __minus__(self, g : 'pose2z') -> 'pose2z':
        return self.between(g)

    def __str__(self):
        return f"pose2z({self._x},{self._y},{self._theta},{self._z})"

    @staticmethod
    def Rz(theta : float) -> np.ndarray:
        return np.array([[np.cos(theta),-np.sin(theta), 0],
                            [np.sin(theta),np.cos(theta), 0],
                            [0,0,1]])
    @staticmethod
    def Rt2T(R : np.ndarray, t : np.ndarray) -> np.ndarray:
        h = np.array([0,0,0,1])
        return np.vstack((np.hstack((R, t)),h))



