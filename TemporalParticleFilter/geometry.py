import numpy as np

class pose2:
    
    size = 3 #state size

    def __init__(self,x,y,theta):
        #worldTx = [wRx , t^w_w->x]
        self.x = x
        self.y = y
        self.theta = theta

    def R(self):
        return np.array([[np.cos(self.theta),-np.sin(self.theta)],
                [np.sin(self.theta),np.cos(self.theta)]])

    def t(self):
        return np.array([[self.x],
                         [self.y]])                   

    def T(self):
        M2x3 = np.hstack([self.R(),self.t()])
        M1x3 = np.array([[0, 0, 1]])
        return np.vstack([M2x3,M1x3])

    def T3d(self):
        T = np.zeros((4,4))
        T[3,3] = 1.0
        T[0:2,3] = self.t().squeeze()
        T[0:2,0:2] = self.R()
        T[2,2] = 1.0
        return T

    def retract(self): #LieAlgebra ExpMap
        return self.T()

    def local(self): #LieAlgebra LogMap
        return np.array([self.x,self.y,self.theta])

    def inverse(self):
        invR = self.R().T
        invt = -invR @ self.t()

        # v = invR[:,0]
        # invtheta = np.arctan2(v[1],v[0])
        invtheta = -self.theta
        return pose2(np.asscalar(invt[0]),
                        np.asscalar(invt[1]),
                        invtheta)

    def transformFrom(self,p: np.ndarray):
        # p - np.array((2,-1))
        # Return point coordinates in global frame.
        return self.R() @ p + self.t()

    def transformTo(self,p : np.ndarray):
        # p - np.array((2,-1))
        # Return world points coordinates in pose coordinate frame
        return self.inverse().transformFrom(p)

    def bearing(self, p : np.ndarray):
        # p - np.array((2,-1))
        # Return angles to p given in world points [-pi,pi]
        p = self.transformTo(p)
        return np.arctan2(p[1,:],p[0,:])

    def range(self, p : np.ndarray):
        # p - np.array((2,-1))
        # Return range of p given inworld points
        p = self.transformTo(p)
        return np.hypot(p[0,:],p[1,:])

    def __add__(self,other):
        #a+b
        #self  = wTa, other = aTb
        #wTb = wTa @ aTb
        wTb = self.T() @ other.T()
        x = wTb[0,2]
        y = wTb[1,2]
        theta = np.arctan2(wTb[1,0],wTb[0,0])
        return pose2(x,y,theta)

    def __sub__(self,other):
        #a-b
        #self = wTa, other = wTb
        #aTb = wTa - wTb = aTw @ wTb
        
        aTb = self.inverse().T() @ other.T()
        x = aTb[0,2]
        y = aTb[1,2]
        theta = np.arctan2(aTb[1,0],aTb[0,0])
        return pose2(x,y,theta)

    def __str__(self):
        return f" x = {self.x}, y = {self.y}, theta = {self.theta}"



