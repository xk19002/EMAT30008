import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from schemes import euler_scheme


def forward_wave_rhs(x,xdiff,v):
    gridx = x.shape[0]
    func = np.empty(gridx)
    func[1:-1] = -v*(x[2:]-x[1:-1])/xdiff
    func[0] = 0
    func[-1] = 0
    return func

v = 1
ld = 1
tend = 0.20
tstep = 0.005
ntstep = int(tend/tstep)
gridx = 101
xdiff = ld/(gridx-1)
xc = np.linspace(0,ld,gridx)
x0 = np.exp(-200*(xc-0.25)**2)
x = np.empty((ntstep+1,gridx))
x[0] = x0.copy()

for k in range(ntstep):
    x[k+1] = euler_scheme(x[k], forward_wave_rhs,tstep,xdiff,v)

fig,ax = plt.subplots(figsize=(10,5))

ax.plot(xc, x[0], label='Initial condition')
ax.plot(xc,x[int(0.10/tstep)],color='purple',label='t=0.10')
ax.plot(xc,x[int(0.15/tstep)],color='red',label='t=0.15')
ax.plot(xc,x[int(tend/tstep)],color='green',label=f't={tend}')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_title('Advection solution using forward Euler scheme and forward finite differences')
ax.legend()
plt.show()

def backward_wave_rhs(x,xdiff,v):
    gridx = x.shape[0]
    func = np.empty(gridx)
    func[1:-1] = -v*(x[1:-1]-x[:-2])/xdiff
    func[0] = 0
    func[-1] = 0
    return func

x = np.empty((ntstep+1,gridx))
x[0] = x0.copy()

for k in range(ntstep):
    x[k+1] = euler_scheme(x[k],backward_wave_rhs,tstep,xdiff,v)

fig,ax = plt.subplots(figsize=(10,5))

ax.plot(xc,x[0],label='Initial condition')
ax.plot(xc,x[int(0.10/tstep)],color='purple',label='t=0.10')
ax.plot(xc,x[int(0.15/tstep)],color='red',label='t=0.15')
ax.plot(xc,x[int(tend/tstep)],color='green',label=f't={tend}')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_title('Advection solution using forward Euler scheme and backward finite differences')
ax.legend()
plt.show()

def periodic_backwward_wave_rhs(x,xdiff,v):
    gridx = x.shape[0]
    func = np.empty(gridx)
    func[1:] = -v*(x[1:]-x[0:-1])/xdiff
    func[0] = func[-1] 
    return func

tend = 2
tstep = 0.002
ntstep = int(tend/tstep)
gridx = 401
xdiff = ld/(gridx-1)
xc = np.linspace(0,ld,gridx)
x0 = np.exp(-200*(xc-0.25)**2)
x = np.empty((ntstep+1,gridx))
x[0] = x0.copy()

for k in range(ntstep):
    x[k+1] = euler_scheme(x[k],periodic_backwward_wave_rhs,tstep,xdiff,v)

fig,ax = plt.subplots(figsize=(10,5))

ax.plot(xc,x[0],label='Initial condition')
ax.plot(xc,x[int(0.68/tstep)],color='red',label='t=0.68')
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_title('Advection solution with periodic boundary conditions')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,5))
graphline, = ax.plot(xc,x0)
timestamp = ax.text(0.05,0.9,'t=0')

ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_title('Advection solution with periodic boundary conditions')
plt.show()

def animation_plot(t):
    q = int(t/tstep)
    graphline.set_ydata(x[q])
    timestamp.set_text(f't={t:.2f}')
    return graphline,timestamp

anim_step = 10
timepoints = np.arange(0,ntstep+anim_step,anim_step)*tstep
anim_plot = animation.FuncAnimation(fig,animation_plot,interval=100,frames=timepoints,repeat=False)
HTML(anim_plot.to_jshtml())
plt.show()