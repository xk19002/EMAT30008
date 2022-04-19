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
ax.loglog(dts,err,'*', color='black',label='Error')
ax.loglog(dts,dts,color='red',label='$dt$')

ax.set_xlabel('$dt$')
ax.set_ylabel('Error')
ax.set_title('Forward Euler accurcy')
ax.legend()
plt.show()

fig,ax = plt.subplots(figsize=(6,6))
stable_reg = plt.Circle((-1,0),1,ec='k',fc='blue',alpha=0.5,hatch='/')

ax.add_artist(stable_reg)
ax.set_aspect(1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

xmin,xmax = -2.3,2.3
ymin,ymax = -2.,2.
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

ax.arrow(xmin,0.,xmax-xmin,0.,fc='k',ec='k',lw=0.5,head_width=1./20.*(ymax-ymin),head_length=1./20.*(xmax-xmin),overhang=0.3,length_includes_head=True,clip_on=False)
ax.arrow(0.,ymin,0.,ymax-ymin, fc='k',ec='k',lw=0.5,head_width=1./20.*(xmax-xmin),head_length=1./20.*(ymax-ymin),overhang=0.3,length_includes_head=True,clip_on=False)

ax.set_xlabel(r'$\lambda_r dt$')
ax.set_ylabel(r'$\lambda_i dt$', rotation=0)
ax.yaxis.set_label_coords(0.6,0.95)
ax.xaxis.set_label_coords(1.05,0.475)
ax.set_xticks((-2,2))
ax.set_yticks((-2,-1,1))
#ax.xaxis.set_major_locator(plt.MaxNLocator(2))
#ax.yaxis.set_major_locator(plt.MaxNLocator(4))
#for i,label in enumerate(ax.yaxis.get_Ticklabels()):
#    if i % 2 != 0 or i == 4:
#        label.set_visible(False)
ax.tick_params(width=2,pad=10)
ax.set_title('Stability region of forward Euler method',y=1.01)
plt.show()

g = 9.81
h_init = 100
v_init = 0

t0 = 0
tend = 4
tstep = 0.1

nstep = int((tend-t0)/tstep)
y = np.empty((nstep+1,2))
y[0] = h_init,v_init

w = np.array([0.,-g])
L = np.array([[0.,1.],[0.,0.]])

for k in range(nstep):
    y[k+1] = y[k] + np.dot(L,y[k])*tstep + w*tstep

tt = np.arange(nstep+1)*tstep

fig,ax = plt.subplots(1,2,figsize=(9,4))

ax[0].plot(tt,y[:,1],'-b',lw=0.8)
ax[0].set_xlim(tt[0],tt[-1])
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$v$')
ax[0].set_title('Speed over time (m/s)')

ax[1].plot(tt,y[:,0],'-b',lw=0.8)
ax[1].set_xlim(tt[0],tt[-1])
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$h$')
ax[1].set_title('Height over time (m)')
plt.show()

spr_k = 2
mass = 1
xinit = 0.75
vinit = 0
tinit = 0
tfinal = 40
ts = 0.15

gama = np.sqrt(spr_k/mass)
ntstep = int((tfinal-tinit)/ts)

yval = np.empty((ntstep+1,2))
yval[0] = xinit,vinit

mat = np.array([[0,1],[-gama**2,0]])

for j in range(ntstep):
    yval[j+1] = yval[j] + np.dot(mat,yval[j])*ts

tval = np.arange(ntstep+1)*ts

fig,axx = plt.subplots(1,2,figsize=(9,4))

axx[0].plot(tval,yval[:,1])
axx[0].set_xlabel('$t$')
axx[0].set_ylabel('$v$')
axx[0].set_title('Speed over time (m/s)')

axx[1].plot(tval,yval[:,0])
axx[1].set_xlabel('$t$')
axx[1].set_ylabel('$x$')
axx[1].set_title('position over time (m)')

for axis in axx:
    axis.set_xlim(0,40)
    
plt.show()