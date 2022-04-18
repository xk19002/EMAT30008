import numpy as np
import matplotlib.pyplot as plt

class forweuler:
    def __init__(self, f, x0, tspan, n):
        self.f, self.x0, self.tspan, self.n = f, x0, tspan, n
        self.dt = tspan/n
        self.x = np.zeros(self.n+1)
        self.t = np.zeros(self.n+1)

    def gen_sol(self):
        self.x[0] = float(self.x0)
        for k in range(self.n):
            self.k = k
            self.t[k+1] = self.t[k] + self.dt
            self.x[k+1] = self.adv_sol()
        return self.x, self.t

    def adv_sol(self):
        x, dt, f, k, t = self.x, self.dt, self.f, self.k, self.t

        xnew = x[k] +dt*f(x[k], t[k])
        return xnew

class forweauler2:
    def __init__(self,f):
        self.f = f

    def init_conds(self,x0):
        self.x0 = float(x0)

    def gen_sol(self, t_steps):
        self.t = np.asarray(t_steps)
        n = len(self.t)
        self.x = np.zeros(n)
        self.x[0] = self.x0

        for k in range(n-1):
            self.k = k
            self.x[k+1] = self.adv_sol()
        return self.x, self.t

    def adv_sol(self):
        x,f,k,t = self.x, self.f, self.k, self.t
        dt = t[k+1] - t[k]
        xnew = x[k] + dt*f(x[k], t[k])
        return xnew

class problem1:
    def __init__(self, c, x0):
        self.c, self.x0 = c, x0

    def __call__(self, x, t):
        return x**3 - x + self.c
        
to_solve = problem1(-2, 0)
method = forweuler(to_solve, to_solve.x0,50,500)
x,t = method.gen_sol()
plt.plot(t,x)
plt.show()

to_solve = problem1(-2,0)
time_span = np.linspace(0,50,501)

method = forweauler2(to_solve)
method.init_conds(to_solve.x0)
x,t = method.gen_sol(time_span)
plt.plot(t,x)
plt.show()