from Lotka_Voltera import hopf_bif
import numpy as np
from scipy import integrate
import math

u10 = 1
u20 = 1
X0 = [u10,u20]
b = 1
sigma = -1
N = 1000
tend = 100
t = np.linspace(0,tend,N)
theta = 1

result = integrate.odeint(hopf_bif,X0,t,args = (b,sigma))

if abs(result - np.sqrt(b)*math.cos(t + theta)) < 1e-6:
    print('Test passed')
else:
    print('Test failed')
