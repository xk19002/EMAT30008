import numpy as np

class ode_solver:

    def __init__(self,f):
        self.f = f

    def init_conds(self,x0):
        if isinstance(x0, (int,float)):
            self.eqns_num = 1
            x0 =float(x0)
        else:
            x0 = np.asarray(x0)
            self.eqns_num = x0.size
        self.x0 = x0

    def solve_ode(self, tstep):
        self.t = np.asarray(tstep)
        tstep_num = self.t.size
        
        self.x = np.zeros((tstep_num, self.eqns_num))
        self.x[0,:] = self.x0

        for k in range(tstep_num-1):
            self.k = k
            self.x[k+1] = self.next_iter()
        return self.x, self.t

    def next_iter(self):
        raise NotImplementedError


        



