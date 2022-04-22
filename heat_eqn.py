import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

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