import numpy as np
import matplotlib.pyplot as plt
import sys
from schemes import euler_scheme

a = 0.1
ld = 1
gridx = 21
xdiff = ld/(gridx -1)
xc = np.linspace(0,ld,gridx)
tinit = 0
tend = 5
fc = 0.49
tstep = fc*xdiff**2/a
ntstep = int((tend-tinit)/tstep)
temp0 = np.sin(2*np.pi*xc)
hs = 2*np.sin(np.pi*xc)

def cent_diff_rhs(temp,xdiff,a,hs):
    gridx = temp.shape[0]
    func = np.empty(gridx)
    func[1:-1] = a/xdiff**2*(temp[:-2]-2*temp[1:-1]+temp[2:])+hs[1:-1]
    func[0] = 0
    func[-1] = 0
    return func

def exactsol(xc,t,a):
    func = (np.exp(-4*np.pi**2*a*t)*np.sin(2*np.pi*xc)+2*(1-np.exp(-np.pi**2*a*t))*np.sin(np.pi*xc)/(np.pi**2*a))
    return func

temp = np.empty((ntstep+1,gridx))
temp[0] = temp0.copy()

for k in range(ntstep):
    temp[k+1] = euler_scheme(temp[k],cent_diff_rhs,tstep,xdiff,a,hs)

fig,ax = plt.subplots(figsize=(10,5))

ax.plot(xc,temp[0],label='Initial condition')
ax.plot(xc,temp[int(0.5/tstep)],color='red',label='$t=0.5$')
ax.plot(xc,temp[-1],color='green',label=f'$t={tend}$')
ax.plot(xc,exactsol(xc,5,a),'b*',label='Exact solution at time $t=5$')
ax.set_xlabel('$x$')
ax.set_ylabel('$Temp.$')
ax.set_title('Heat transport using forward Euler and forward finite differences')
ax.legend()
plt.show()

def l2_norm_diff(func1,func2):
    ldiff = np.sqrt(np.sum((func1-func2)**2))/func1.shape[0]
    return ldiff

err = l2_norm_diff(temp[-1],exactsol(xc,tend,a))
print(f'L2-error: {err}')

conv_prec = 1e-6
a = 0.1
ld = 1
tinit = 0
tend = 5
fc = 0.49
max_refine = 8
refine = 1
gridx = 3
sol_ref = np.zeros(gridx)
soldiff = 1

while(soldiff > conv_prec):
    if refine > max_refine:
        print('\n Solution did not converge within the desired maximum grid refinements')
        print(f'Grid points tested: gridx = {gridx}')
        break

    gridx = 2*gridx-1
    xdiff = ld/(gridx-1)
    xc = np.linspace(0,ld,gridx)
    tempinit = np.sin(2*np.pi*xc)
    hs = 2*np.sin(np.pi*xc)
    tstep = fc*xdiff**2/a
    ntstep = int((tend-tinit)/tstep)

    for k in range(ntstep):
        temp1 = euler_scheme(tempinit,cent_diff_rhs,tstep,xdiff,a,hs)
        tempinit = temp1

    soldiff = l2_norm_diff(temp1[::2],sol_ref)
    print(f'L2-norm difference after {refine} grid refinements'f' (number of grid points = {gridx}): {soldiff}')

    refine += 1
    sol_ref = temp1.copy()

else:
    print(f'\nConvergence of solution within required precision'f' for {gridx} grid points')





