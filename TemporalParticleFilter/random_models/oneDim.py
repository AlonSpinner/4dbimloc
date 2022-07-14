from re import A
import numpy as np
import matplotlib.pyplot as plt
from math import erf


class gaussian1D:
    def __init__(self,mu : float,sigma : float) -> None:
        self.mu : float = mu
        self.sigma : float = sigma

    def pdf(self,x : float):
        return 1/(np.sqrt(2*np.pi)*self.sigma) * np.exp(-(x-self.mu)**2/(2*self.sigma**2))
        
    def sample(self, n : int = 1):
        return np.random.normal(self.mu, self.sigma, n)

    def cdf(self,x : float): #cumulative distibution function
        # https://en.wikipedia.org/wiki/Normal_distribution
        #returns integral from [-inf,x]
        zeta = (x-self.mu)/(self.sigma)
        return npPhi(zeta)

    def plot(self) -> None:
        dt = 0.1
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

class gaussian1DT:
#https://en.wikipedia.org/wiki/Truncated_normal_distribution
    def __init__(self,mu,sigma,a,b):
        self.mu : float = mu
        self.sigma : float = sigma
        self.a = a
        self.b = b

        self.alpha = (a-mu)/sigma
        self.beta = (b-mu)/sigma
        self.Z = npPhi(self.beta) - npPhi(self.alpha)

    def pdf(self,x):
        zeta = (x-self.mu)/self.sigma
        phi = 1/np.sqrt(2*np.pi) * np.exp(-0.5*zeta**2) * ((self.a < x) & (x < self.b))
        return phi/(self.sigma*self.Z)

    def cdf(self,x):
        zeta = (x-self.mu)/self.sigma
        return (npPhi(zeta)-npPhi(self.alpha))/self.Z * ((self.a < x) & (x < self.b)) + 1.0 * (self.b <= x)

    def sample(self, n : int = 1):
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

    def plot(self):
        dt = 0.1
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

class uniform1D:
    def __init__(self,a : float, b : float) -> None:
        self.a : float = a
        self.b : float = b
        self.l : float = b-a

    def pdf(self,x : float):
            return 1/self.l * ((self.a < x) & (x < self.b))

    def sample(self, n : int = 1):
        return np.random.uniform(self.a, self.b, n)

    def cdf(self,x : float): #cumulative distibution function
        # https://en.wikipedia.org/wiki/Normal_distribution
        #returns integral from [-inf,x]
        return (x-self.a)/self.l * ((self.a < x) & (x < self.b)) + 1.0 * (self.b <= x)

    def plot(self) -> None:
        dt = 0.1
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

class dead1D:
    def __init__(self, status : str = 'free') -> None:
        self.status = status

    def pdf(self,x : float):
        return np.zeros_like(x)

    def sample(self, n : int = 1): #sample time of construction
        if self.status == 'free':
            return np.ones(n) * 1e10 
        elif self.status == 'occ':
            return np.ones(n) * -1e10 

    def cdf(self,x : float):
        if self.status == 'free':
            return np.zeros_like(x)
        elif self.status == 'occ':
            return np.ones_like(x)

def nperf(x):
    if np.isscalar(x):
        return erf(x)
    else:
        return np.array([erf(val) for val in x])

def npPhi(x):
    return 0.5*(1 + nperf(x/np.sqrt(2)))



    