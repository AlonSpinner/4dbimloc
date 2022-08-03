import numpy as np

class Pose2z:    
    state_size : int = 4 #state size

    def __init__(self, x, y, theta, z) -> None:
        #we hold the storage form
        self._x : float = x
        self._y : float = y
        self._theta : float = theta
        self._z : float = z
        
        self._R  : np.ndarray = self.Rz(self._theta)
        self._t : np.ndarray = np.vstack((self._x, self._y, self._z))

    #prevents users from settings internal storage directly. Basically creating an immutable class
         #example in: https://realpython.com/python-property/
    @property
    def x(self) -> float:
        return self._x
    @property
    def y(self) -> float:
        return self._y
    @property
    def theta(self) -> float:
        return self._theta
    @property
    def z(self) -> float:
        return self._z
    @property
    def R(self) -> float:
        return self._R
    @property
    def t(self) -> float:
        return self._t

    @classmethod
    def from_Exp(cls, T : np.ndarray) -> 'Pose2z':
        #T is a 4x4 transformation matrix
        x = T[0,3]; y = T[1,3]; z = T[2,3]
        theta = np.arctan2(T[1,0], T[0,0])
        return cls(x ,y ,theta, z)

    @classmethod
    def from_Rt(cls, R : np.ndarray, t : np.ndarray) -> 'Pose2z':
        h = np.array([0,0,0,1])
        return cls.from_Exp(cls.Rt2T(R,t))

    @staticmethod
    def identity() -> 'Pose2z':
        return Pose2z(0,0,0,0)

    def Exp(self) -> np.ndarray:
        return self.Rt2T(self._R,self._t)

    def Log(self) -> np.ndarray:
        #slow but correct as opposed to just returnning internal storage.. whatevah
        theta = np.arctan2(self._R[1,0], self._R[0,0])
        return np.array([self._t[0,0], self._t[1,0], theta, self._t[2,0]])

    def compose(self, g : 'Pose2z') -> 'Pose2z':
        #returns the composition of self and x.
        return self.from_Exp(self.Exp() @ g.Exp())

    def inverse(self) -> 'Pose2z':
        #returns the inverse of self.
        invR = self._R.T
        invt = -invR @ self._t
        return self.from_Rt(invR,invt)

    def between(self, g : 'Pose2z') -> 'Pose2z':
        return self.compose(self.inverse(),g)

    def transform_to(self, p : np.ndarray) -> np.ndarray:
        return self.Exp() @ p

    def transform_from(self, p : np.ndarray) -> np.ndarray:
        return self.inverse().Exp() @ p

    def localCoordinates(self, x : 'Pose2z') -> 'Pose2z':
        return self.between(x).Log()

    def retract(self, v : np.ndarray) -> 'Pose2z':
        return self.compose(Pose2z(*v))

    def __mul__(self, x : 'Pose2z') -> 'Pose2z':
        return self.compose(x)
    
    def __add__(self, v : np.ndarray) -> 'Pose2z':
        return self.retract(v)

    def __minus__(self, g : 'Pose2z') -> 'Pose2z':
        return self.between(g)

    def __str__(self):
        return f"Pose2z({self._x},{self._y},{self._theta},{self._z})"

    @staticmethod
    def Rz(theta : float) -> np.ndarray:
        return np.array([[np.cos(theta),-np.sin(theta), 0],
                            [np.sin(theta),np.cos(theta), 0],
                            [0,0,1]])
    @staticmethod
    def Rt2T(R : np.ndarray, t : np.ndarray) -> np.ndarray:
        h = np.array([0,0,0,1])
        return np.vstack((np.hstack((R, t)),h))



