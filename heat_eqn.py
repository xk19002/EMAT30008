import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import inv

gridx = 41
intl = 1
xdiff = intl/(gridx - 1)
x = np.linspace(0,1,gridx)
bvec = -1*np.ones(gridx)
temp = np.empty(gridx)

def dirichlet_heat_mat(gridx,xdiff):
    mat_diags = [[1],[-2],[1]]
    offset_pos = [-1,0,1]
    dir_mat = diags(mat_diags,offset_pos,shape=(gridx-2,gridx-2)).toarray()
    return dir_mat/xdiff**2

mat = dirichlet_heat_mat(gridx,xdiff)
print(mat)

inv_mat = inv(mat)
temp[1:-1] = np.dot(inv_mat,bvec[1:-1])
temp[0],temp[-1] = [0,0]

exact_temp = 0.5*x*(1-x)

fig,ax = plt.subplots(figsize=(10,7))

ax.plot(x,exact_temp,label='Exact solution')
ax.plot(x,temp,'ro',label='Approximate solution')
ax.set_xlabel('$x$')
ax.set_ylabel('$T$')
ax.set_title('Heat equation with homogeneous Dirichlet boundary conditions')
ax.legend()
plt.show()

bvec[1] = bvec[1] - 1/xdiff**2
temp[1:-1] = np.dot(inv_mat,bvec[1:-1])
temp[0],temp[-1] = [1,0]
exact_temp = 0.5*(x+2)*(1-x)

fig,ax = plt.subplots(figsize=(10,7))

ax.plot(x,exact_temp,label='Exact solution')
ax.plot(x,temp,'ro',label='Approximate solution')
ax.set_xlabel('$x$')
ax.set_ylabel('$T$')
ax.set_title('Heat equation with non-homogeneous Dirichlet boundary conditions')
ax.legend()
plt.show()

gridx = 41
intl = 1
xdiff = intl/(gridx - 1)
x = np.linspace(0,1,gridx)
bvec = -1*np.ones(gridx)
temp = np.empty(gridx)

mat = dirichlet_heat_mat(gridx,xdiff)
mat[0,0:2] = np.array([-2/3,2/3])/xdiff**2
inv_mat = inv(mat)

bvec[1] = bvec[1] + 4/(3*xdiff)
bvec[-2] = bvec[-2] - 1/xdiff**2

temp[1:-1] = np.dot(inv_mat,bvec[1:-1])
temp[0] = 4/3*temp[1] - 1/3*temp[2] - 4/3*xdiff
temp[-1] = 1
exact_temp = -0.5*(x**2-4*x+1)

fig,ax =plt.subplots(figsize=(10,7))

ax.plot(x,exact_temp,label='Exact solution')
ax.plot(x,temp,'ro',label='Approximate solution')
ax.set_xlabel('$x$')
ax.set_ylabel('$T$')
ax.set_title('Heat equation with non-homogeneous Neumann boundary conditions')
ax.legend()
plt.show()