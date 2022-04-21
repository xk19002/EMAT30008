import numpy as np
import matplotlib.pyplot as plt

g = 9.81
hinit = 100
vinit = 0
tinit = 0
tend = 4
dt = 0.1

nts = int((tend - tinit)/dt)
y = np.empty((nts+1,2))

y[0] = hinit,vinit

a = np.array([0,-g])
M = np.array([[0,1],[0,0]])

for k in range(nts):
    y_val = y[k] + 0.5*dt*(np.dot(M, y[k])+a)
    y[k+1] = y[k] + dt*(np.dot(M,y_val)+a)

t = np.arange(nts+1)*dt

fig,ax = plt.subplots(1,2,figsize=(9,4))

ax[0].plot(t,y[:,1])
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$v$')
ax[0].set_title('Speed over time (m/s)')

ax[1].plot(t,y[:,0])
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$h$')
ax[1].set_title('Height over time (m)')

for axx in ax:
    axx.set_xlim(t[0],t[-1])

plt.show()