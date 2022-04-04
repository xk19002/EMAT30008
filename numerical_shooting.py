import numpy as np
import math
import matplotlib.pyplot as plt

h=0.01
lower_bound = 0
upper_bound = 10

N = int(upper_bound - lower_bound/h)
b = 1
sigma = -1
theta = 1
w1 = np.zeros(N)
w2 = np.zeros(N)
u1 = np.zeros(N)
u2 = np.zeros(N)
analytic_sol1 = np.zeros(N)
analytic_sol2 = np.zeros(N)

w1[0] = 0
w2[0] = 0
u1[0] = 0
u2[0] = 0
analytic_sol1[0] = 0
analytic_sol2[0] = 0

for i in range (1,N):
    w1[i] = w1[i-1] + h * (b*u1[i-1] - u2[i-1] + sigma*u1[i-1]*((u1[i-1])^2 + (u2[i-1])^2))
    w2[i] = w2[i-1] + h * (u1[i-1] + b*u2[i-1] + sigma*u2[i-1]*((u1[i-1])^2 + (u2[i-1])^2)) 
    u1[i] = u1[i-1] + h
    u2[i] = u2[i-1] + h
    analytic_sol1[i] = np.sqrt(b) * math.cos(i + theta)
    analytic_sol2[i] = np.sqrt(b) * math.sin(i + theta)