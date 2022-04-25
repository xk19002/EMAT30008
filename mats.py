import numpy as np
from scipy.sparse import diags

def dirichlet_bound_mat(gridx,xdiff):
    mat_diags = [[1],[-2],[1]]
    offset_pos = [-1,0,1]
    dir_mat = diags(mat_diags,offset_pos,shape=(gridx-2,gridx-2)).toarray()
    return dir_mat/xdiff**2
