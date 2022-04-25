import numpy as np

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

