import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,1,1000)

def ferguson(t,p1,dp1,p2,dp2):
    a0 = p1
    a1 = dp1
    a2 = 3*(p2-p1) - 2*dp1 - dp2
    a3 = 2*(p1-p2) + dp1 + dp2
    return a0 + a1*t + a2*t**2 + a3*t**3

def q_ferguson(t,p0,dp0,p1,dp1,q):
    #https://www.researchgate.net/publication/328786937_q_-Ferguson_curves/fulltext/5be3012092851c6b27ada047/q-Ferguson-curves.pdf
    a0 = p0
    a1 = dp0
    a2 = ((q**2+q+1)*(p1-p0) - (q**2+q)*dp0 - dp1)/q**2
    a3 = ((q+1)*(p0-p1) + q*dp0 + dp1)/q**2
    return a0 + a1*t + a2*t**2 + a3*t**3

q1 = 0
q2 = 1
dq1 = 4
dq2 = 1

plt.plot(t, ferguson(t,q1,dq1,q2,dq2))
plt.plot(t, q_ferguson(t,q1,dq1,q2,dq2,0.8))
plt.plot(t, np.power(t,0.2))
plt.show()
plt.axis('equal')
plt.grid(True)
