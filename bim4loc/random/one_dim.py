import numpy as np
import matplotlib.pyplot as plt
from math import erf
from typing import Tuple
from numba import njit

class Distribution1D:
    def sample(n : int) -> np.ndarray:
        pass
    def pdf(x : np.ndarray) -> np.ndarray:
        pass
    def cdf(x : np.ndarray) -> np.ndarray:
        pass
    def plot() -> Tuple[plt.Figure, plt.Axes]:
        pass

class Gaussian(Distribution1D):
    def __init__(self,mu : float,sigma : float) -> None:
        self.mu : float = mu
        self.sigma : float = sigma

    def sample(self, n : int = 1) -> np.ndarray:
        return np.random.normal(self.mu, self.sigma, n)

    def pdf(self,x : np.ndarray) -> np.ndarray:
        return self._pdf(self.mu,self.sigma,x)

    def cdf(self,x : np.ndarray) -> np.ndarray: #cumulative distibution function
        # https://en.wikipedia.org/wiki/Normal_distribution
        #returns integral from [-inf,x]
        zeta = (x-self.mu)/(self.sigma)
        return npPhi(zeta)

    def plot(self, dt = 0.1) -> Tuple[plt.Figure, plt.Axes]:
        tmin = self.mu - 3 * self.sigma
        tmax = self.mu + 3 * self.sigma
        t = np.array(np.arange(tmin,tmax,dt))
        dydt = self.pdf(t)
        y = self.cdf(t)

        fig, axes = plt.subplots(1,2)
        axes[0].plot(t,y)
        axes[0].set_title('cdf(t)')
        axes[1].plot(t,dydt)
        axes[1].set_title('pdf(t)')
        return fig,axes

    @staticmethod
    @njit()
    def _pdf(mu : float ,sigma : float, x : np.ndarray, pseudo = False) -> np.ndarray:
        num = np.exp(-(x-mu)**2/(2*sigma**2))
        if pseudo:
            return num
        else:
            den = 1/(np.sqrt(2*np.pi)*sigma)
            return num/den

class GaussianT(Distribution1D):
#https://en.wikipedia.org/wiki/Truncated_normal_distribution
    def __init__(self,mu,sigma,a,b):
        self.mu : float = mu
        self.sigma : float = sigma
        self.a = a
        self.b = b

        self.alpha = (a-mu)/sigma
        self.beta = (b-mu)/sigma
        self.Z = npPhi(self.beta) - npPhi(self.alpha)

    def sample(self, n : int = 1) -> np.ndarray:
        #algorithm from https://arxiv.org/pdf/0907.4010.pdf
        
        #normalize
        na = (self.a-self.mu)/self.sigma
        nb = (self.b-self.mu)/self.sigma

        k = 0
        samples = np.zeros(n)
        while k < n:
            z = np.random.uniform(na,nb)
            
            if na <= 0 <= nb:
                pz = np.exp((-z**2)/2)
            elif nb < 0:
                pz = np.exp((self.b**2-z**2)/2)
            elif 0 < na:
                pz = np.exp((self.a**2-z**2)/2)
            
            u = np.random.uniform(0,1)
            if u <= pz:
                samples[k] = z*self.sigma + self.mu #take z and unnormalize it
                k += 1
                continue
        
        return samples

    def pdf(self,x : np.ndarray) -> np.ndarray:
        zeta = (x-self.mu)/self.sigma
        phi = 1/np.sqrt(2*np.pi) * np.exp(-0.5*zeta**2) * ((self.a < x) & (x < self.b))
        return phi/(self.sigma*self.Z)

    def cdf(self,x : np.ndarray) -> np.ndarray:
        zeta = (x-self.mu)/self.sigma
        return (npPhi(zeta)-npPhi(self.alpha))/self.Z * ((self.a < x) & (x < self.b)) + 1.0 * (self.b <= x)

    def plot(self, dt = 0.1) -> Tuple[plt.Figure, plt.Axes]:
        tmin = self.a - self.sigma
        tmax = self.b + self.sigma
        t = np.array(np.arange(tmin,tmax,dt))
        dydt = self.pdf(t)
        y = self.cdf(t)

        fig, axes = plt.subplots(1,2)
        axes[0].plot(t,y)
        axes[0].set_title('cdf(t)')
        axes[1].plot(t,dydt)
        axes[1].set_title('pdf(t)')
        return fig,axes

class Uniform(Distribution1D):
    def __init__(self,a : float, b : float) -> None:
        self.a : float = a
        self.b : float = b
        self.l : float = b-a

    def sample(self, n : int = 1) -> np.ndarray:
        return np.random.uniform(self.a, self.b, n)

    def pdf(self,x : np.ndarray) -> np.ndarray:
            return 1/self.l * ((self.a < x) & (x < self.b))

    def cdf(self,x : float): #cumulative distibution function
        # https://en.wikipedia.org/wiki/Normal_distribution
        #returns integral from [-inf,x]
        return (x-self.a)/self.l * ((self.a < x) & (x < self.b)) + 1.0 * (self.b <= x)

    def plot(self, dt = 0.1) -> Tuple[plt.Figure, plt.Axes]:
        tmin = self.a - self.l/2
        tmax = self.b + self.l/2
        t = np.array(np.arange(tmin,tmax,dt))
        dydt = self.pdf(t)
        y = self.cdf(t)

        fig, axes = plt.subplots(1,2)
        axes[0].plot(t,y)
        axes[0].set_title('cdf(t)')
        axes[1].plot(t,dydt)
        axes[1].set_title('pdf(t)')
        return fig,axes


#----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- ASSISTING FUNCTIONS -----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

def nperf(x : np.ndarray) -> np.ndarray:
    if np.isscalar(x):
        return erf(x)
    else:
        return np.array([erf(val) for val in x])

def npPhi(x : np.ndarray) -> np.ndarray:
    return 0.5*(1 + nperf(x/np.sqrt(2)))




    