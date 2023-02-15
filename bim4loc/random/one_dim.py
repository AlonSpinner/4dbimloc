import numpy as np
import matplotlib.pyplot as plt
from math import erf
from typing import Tuple
from numba import njit, prange
from scipy import stats

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

    def Anderson_Darling(self, v):
        #https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test
        v = np.sort(v)
        n = len(v)
        v = (v - self.mu) /self.sigma
        s = 0
        for i in range(n):
            s += (2*i -1)*np.log(npPhi(v[i])) + (2*(n-i) + 1)*np.log(1-npPhi(v[i]))
        A2 = -n - s/n
        A2_star = A2*(1 + 0.75/n + 2.25/n**2)
        return A2_star

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
    @njit(cache = True)
    def _pdf(mu : np.ndarray ,sigma : float, x : np.ndarray, pseudo = False) -> np.ndarray:
        num = np.exp(-(x-mu)**2/(2*sigma**2))
        if pseudo:
            return num
        den = np.sqrt(2*np.pi)*sigma
        return num / den

    @staticmethod
    def Shapiro_Wilk(v):
        #https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
        stat, p_value = stats.shapiro(v)
        return p_value

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
        
        return samples

    def pdf(self,x : np.ndarray) -> np.ndarray:
        zeta = (x-self.mu)/self.sigma
        phi = 1/np.sqrt(2*np.pi) * np.exp(-0.5*zeta**2) * ((self.a < x) & (x < self.b))
        return phi/(self.sigma*self.Z)

    @staticmethod
    @njit(cache = True)
    def _pdf(mu : np.ndarray ,sigma : float, x : np.ndarray, a,b) -> np.ndarray:
        alpha = (a-mu)/sigma
        beta = (b-mu)/sigma
        Z = npPhi(beta) - npPhi(alpha)
        zeta = (x-mu)/sigma
        phi = 1/np.sqrt(2*np.pi) * np.exp(-0.5*zeta**2) * ((a < x) & (x < b))
        return phi/(sigma*Z)

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

class ExponentialT(Distribution1D):
    #from probablistic robotics (Thrun, Burgard, Fox) p. 154
    #chapter 6.3  - Beam Models of Range Finders
    def __init__(self, lamBda : float, maxX : float) -> None:
        self.lamBda : float = lamBda
        self.maxX = maxX

    def sample(self, n : int = 1) -> np.ndarray:
        #did not verify this
        samples = np.zeros(n)
        k = 0
        while k < n:
            x =  np.random.exponential(1/self.lamBda)
            if x < self.maxX:
                samples[k] = x
                k += 1
        return samples

    def pdf(self,x : np.ndarray) -> np.ndarray:
            return self._pdf(self.lamBda, self.maxX, x)

    def cdf(self,x : float): #cumulative distibution function
        #returns integral from [-inf,x]
        return (1.0-np.exp(-self.lamBda*x)) / (1.0-np.exp(-self.lamBda*self.maxX))

    def plot(self, dt = 0.1) -> Tuple[plt.Figure, plt.Axes]:
        tmin = 0
        tmax = self.maxX
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
    @njit(cache = True)
    def _pdf(lamBda : float, maxX : float, x : np.ndarray) -> np.ndarray:
        return (lamBda * np.exp(-lamBda*x))/(1-np.exp(-lamBda*maxX))  * (x <= maxX)

#----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- ASSISTING FUNCTIONS -----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
@njit(cache = True)
def erf_approx(x):
    #from https://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python
    #error < 1e-7
    # save the sign of x
    sign = np.sign(x)
    x = np.abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

@njit(cache = True)
def nperf(x : np.ndarray) -> np.ndarray:
    return erf_approx(x)
    if np.isscalar(x):
        return erf(x)
    else:
        for i in prange(x.size):
            x[i] = erf(x[i])
        return x

@njit(cache = True)
def npPhi(x : np.ndarray) -> np.ndarray:
    return 0.5*(1 + nperf(x/np.sqrt(2)))




    