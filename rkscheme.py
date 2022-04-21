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

resx = 100
resy = 100

x = np.linspace(-3.5,1.5,resx)
y = np.linspace(-3.5,3.5,resy)

gridx, gridy = np.meshgrid(x,y)
z = gridx + 1j*gridy

rem1 = 1 + z
norm_rem1 = np.real(rem1*rem1.conj())
rem2 = 1 + z + z**2/2
norm_rem2 = np.real(rem2*rem2.conj())
rem4 = 1 + z + z**2/2 + z**3/6 + z**4/24
norm_rem4 = np.real(rem4*rem4.conj())

fig,ax = plt.subplots(figsize=(8,8))

ax.contour(gridx,gridy,norm_rem1,levels=[1],colors='b')
ax.contour(gridx,gridy,norm_rem2,levels=[1],colors='g')
ax.contour(gridx,gridy,norm_rem4,levels=[1],colors='r')

ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

xlo,xhi = -3.2,1.4
ylo,yhi = -3.4,3.4

ax.set_xlim(xlo,xhi)
ax.set_ylim(ylo,yhi)

ax.arrow(xlo,0,xhi-xlo,0,fc='k',ec='k',lw=0.5,head_width=1/20*(yhi-ylo),head_length=1/20*(xhi-xlo),overhang=0.3,length_includes_head=True,clip_on=False)
ax.arrow(0,ylo,0,yhi-ylo,fc='k',ec='k',lw=0.5,head_width=1/20*(xhi-xlo),head_length=1/20*(yhi-ylo),overhang=0.3,length_includes_head=True,clip_on=False)

ax.yaxis.set_label_coords(0.85,0.95)
ax.xaxis.set_label_coords(1.05,0.475)

ax.set_xticks((-3.-1,1))
ax.set_yticks((-2,-1,1,2))

ax.text(-1,1.1,r'Euler',fontsize=14,horizontalalignment='center')
ax.text(-1,1.85,r'RK2',fontsize=14,horizontalalignment='center')
ax.text(-2.05,2.05,r'RK4',fontsize=14,horizontalalignment='center')

ax.arrow(0.5,2.63,-0.5,0.2,fc='k',ec='k',lw=0.5,head_width=1/80*(yhi-ylo),head_length=1/50*(xhi-xlo),overhang=0.3,length_includes_head=True,clip_on=False)
ax.text(0.75,2.55,r'2.83',fontsize=14,horizontalalignment='center')

ax.arrow(-3.05,0.4,0.26,-0.4,fc='k',ec='k',lw=0.5,head_width=1/80*(yhi-ylo),head_length=1/50*(xhi-xlo),overhang=0.3,length_includes_head=True,clip_on=False)
ax.text(-3.17,0.49,r'-2.79',fontsize=14,horizontalalignment='center')

ax.set_xlabel(r'$\lambda_r dt$')
ax.set_ylabel(r'$\lambda_i dt$',rotation=0)

ax.set_aspect(1)
ax.set_title('Stability regions for Euler, RK2, RK4 schemes',x=0.7,y=1.01)
plt.show()

g = 9.81
h_init = 100
v_init = 0
t_init = 0
t_end = 10
t_diff = 0.5

n_tsteps = int((t_end - t_init)/t_diff)

exp_y = np.empty((n_tsteps+1,2))
imp_y = np.empty((n_tsteps+1,2))

exp_y[0] = h_init,v_init
imp_y[0] = h_init,v_init

w = np.array([0,-g])
m = np.array([[0,1],[0,0]])

k_mat = np.linalg.inv(np.eye(2)-m*t_diff/4)

for k in range(n_tsteps):
    yval = exp_y[k] + 0.5*t_diff*(np.dot(m,exp_y[k])+w)
    exp_y[k+1] = exp_y[k] + t_diff*(np.dot(m,yval)+w)

    k1 = np.dot(k_mat,np.dot(m,imp_y[k])+w)
    k2 = np.dot(k_mat,np.dot(m,imp_y[k]+k1*t_diff/2)+w)
    imp_y[k+1] = imp_y[k] + 0.5*t_diff*(k1+k2)

tval = np.arange(n_tsteps+1)*t_diff

fig,ax = plt.subplots(1,2,figsize=(9,4))

ax[0].plot(tval, exp_y[:,0],'--')
ax[0].plot(tval, imp_y[:,0],'-')
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$v$')
ax[0].set_title('Speed over time (m/s)')

ax[1].plot(tval,exp_y[:,1],'--')
ax[1].plot(tval,imp_y[:,1],'-')
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$h$')
ax[1].set_title('Height over time (m)')

for axis in ax:
    axis.set_xlim(tval[0], tval[-1])

plt.show()