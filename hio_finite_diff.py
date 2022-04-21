import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
gridx = 200
intl = pi
xdiff = intl/(gridx-1)

xval = np.linspace(0,intl,gridx)
func = np.exp(xval)*np.sin(3*pi*xval)
deriv2 = np.exp(xval)*(np.sin(3*pi*xval)+6*pi*np.cos(3*pi*xval)-9*pi**2*np.sin(3*pi*xval))

gridx = 80
xdiff = intl/(gridx-1)

xval_c = np.linspace(0,intl,gridx)
func_c = np.exp(xval_c)*np.sin(3*pi*xval_c)
deriv2_c = np.empty(gridx)
deriv2_c[1:-1] = (func_c[:-2]-2*func_c[1:-1]+func_c[2:])/xdiff**2

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(xval[1:-1], deriv2[1:-1])
ax.plot(xval_c[1:-1],deriv2_c[1:-1],'ro')
ax.set_xlabel('$x$')
ax.set_ylabel('$f\'$')
plt.show()
