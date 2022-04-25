from timeit import timeit
import numpy as np
import matplotlib.pyplot as plt
import sys
from schemes import euler_scheme
from err_norms import l2_norm_diff
from mats import dirichlet_bound_mat
from pde_mod import cent_diff_rhs, exactsol

a = 0.1
ld = 1
tinit = 0
tend = 5
gridx = 513
xdiff = ld / (gridx-1)
xc = np.linspace(0,ld,gridx)
fc = 0.49
tstep = fc*xdiff**2/a
ntstep = int((tend-tinit)/tstep)
temp0 = np.sin(2*np.pi*xc)
hs = 2*np.sin(np.pi*xc)
exsol = exactsol(xc,tend,a)
temp = np.empty((ntstep+1,gridx))
temp[0] = temp0.copy()


for k in range(ntstep):
    temp[k+1] = euler_scheme(temp[k], cent_diff_rhs,tstep,xdiff,a,hs)

l2_err = l2_norm_diff(temp[-1],exsol)
print(f'L2-error: {l2_err}')
print(f'Steps required for time integration: {ntstep}')

dirmat = dirichlet_bound_mat(gridx,xdiff)
mat = np.eye(gridx-2) + a*tstep*dirmat
temp[0] = temp0.copy()

for k in range(ntstep):
    temp[k+1,1:-1] = np.dot(mat,temp[k,1:-1]) + hs[1:-1]*tstep

temp[-1,0] = 0
temp[-1,-1] = 0

l2_err = l2_norm_diff(temp[-1],exsol)
print(f'L2-error: {l2_err}')
print(f'Steps required for time integration: {ntstep}')
