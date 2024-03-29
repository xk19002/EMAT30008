import numpy as np
from tqdm import tqdm

class solve_ode:

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

    def solve_to(self, tstep):
        self.t = np.asarray(tstep)
        tstep_num = self.t.size
        
        self.x = np.zeros((tstep_num, self.eqns_num))
        self.x[0,:] = self.x0

        for k in tqdm(range(tstep_num-1), ascii=True):
            self.k = k
            self.x[k+1] = self.next_iter()
        return self.x, self.t

    def next_iter(self):
        raise NotImplementedError

class euler_step(solve_ode):
    def next_iter(self):
        x,f,k,t = self.x, self.f, self.k, self.t
        step_size = t[k+1] - t[k]
        return x[k,:] + step_size * f(x[k,:], t[k])

class RK4_step(solve_ode):
    def next_iter(self):
        x,f,k,t = self.x, self.f, self.k, self.t
        step_size = t[k + 1] - t[k]
        half_step_size = step_size/2
        sol1 = step_size * f(x[k,:], t[k])
        sol2 = step_size *f(x[k,:] + 0.5 * sol1, t[k] + half_step_size)
        sol3 = step_size *f(x[k,:] + 0.5 * sol2, t[k] + half_step_size)
        sol4 = step_size *f(x[k,:] + sol3, t[k] + half_step_size)
        return x[k,:] + (1/6) * (sol1 + 2 * sol2 + 2 * sol3 + sol4)

    
        



