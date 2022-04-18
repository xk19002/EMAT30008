import numpy as np
import matplotlib.pyplot as plt

class solve_ODE:
    def __init__(self,f):
        self.f = f

    def adv_sol(self):
        raise NotImplementedError

    def init_conds(self,x0):
        self.x0 = float(x0)

    def sol_meth(self, tsteps):
        self.t = np.asarray(tsteps)
        n = len(self.t)
        self.x = np.zeros(n)
        self.x[0] = self.x0

        for k in range(n-1):
            self.k = k
            self.x[k+1] = self.adv_sol()
        return self.x, self.t

class forweuler(solve_ODE):
    def adv_sol(self):
        x,f,k,t = self.x,self.f,self.k,self.t

        dt = t[k+1] - t[k]
        xnew = x[k] + dt*f(x[k],t[k])
        return xnew

class midpoint(solve_ODE):
    def adv_sol(self):
        x,f,k,t = self.x, self.f, self.k, self.t
        dt = t[k+1] - t[k]
        dt2 = dt/2
        k1 = f(x[k], t)
        k2 = f(x[k] + dt2*k1, t[k] + dt2)
        xnew = x[k] +dt*k2
        return xnew

class RK4(solve_ODE):
    def adv_sol(self):
        x,f,k,t = self.x,self.f,self.k,self.t
        dt = t[k+1] - t[k]
        dt2 = dt/2.0
        k1 = f(x[k], t)
        k2 = f(x[k] + dt2*k1, t[k] + dt2)
        k3 = f(x[k] + dt2*k2, t[k] + dt2)
        k4 = f(x[k] + dt*k3, t[k] + dt)
        xnew = x[k] + (dt/6.0)*(k1 + 2*k2 + 2*k3 +k4)
        return xnew

