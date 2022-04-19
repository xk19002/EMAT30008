import numpy as np
import matplotlib.pyplot as plt

coeff = 0.25
t_init = 0
t_final = 5
dt = 0.5
n_init = 100

n_steps = int((t_final - t_init)/dt)
n = np.empty(n_steps + 1)
n[0] = n_init

for k in range(n_steps):
    n[k+1] = n[k] - coeff*n[k]*dt

t = np.arange(n_steps+1)*dt

exact_sol = n_init * np.exp(-coeff*t)

fig, ax = plt.subplots()
ax.plot(t,exact_sol,linestyle='-',label='Exact solution')
ax.plot(t,n,'ro',label='Forward Euler')
ax.set_xlabel('$t$')
ax.set_ylabel('$n$')
ax.set_title('Radioactive decay')
ax.legend()
plt.show()

dts = np.array([0.5/2**i for i in range(5)])
val_list = np.empty_like(dts)

for j,dt in enumerate(dts):
    n = n_init
    n_steps = int((t_final-t_init)/dt)

    for i in range(n_steps):
        n = n - coeff*n*dt
    
    val_list[j] = n

err = np.abs(val_list-n_init*np.exp(-coeff*t_final))

fig,ax = plt.subplots()
ax.loglog(dts,err,'*', color='red',label='Error')
ax.loglog(dts,dts,color='red',label='$dt$')

ax.set_xlabel('$dt$')
ax.set_ylabel('Error')
ax.set_title('Forward Euler accurcy')
ax.legend()
plt.show()


