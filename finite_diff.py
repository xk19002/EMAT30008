import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
gridx = 200
intl = pi
xdiff = intl/(gridx-1)

xval = np.linspace(0,intl,gridx)
func = np.exp(xval)*np.sin(3*pi*xval)
deriv = np.exp(xval)*(np.sin(3*pi*xval)+3*pi*np.cos(3*pi*xval))

gridx = 80
intl = pi
xdiff = intl/(gridx-1)

xval_c = np.linspace(0,intl,gridx)
func_c = np.exp(xval_c)*np.sin(3*pi*xval_c)

for_diff = np.empty(gridx)
back_diff = np.empty(gridx)
cent_diff = np.empty(gridx)

for k in range(0,gridx-1):
    for_diff[k] = (func_c[k+1] -func_c[k])/xdiff

for k in range(1,gridx):
    back_diff[k] = (func_c[k] - func_c[k-1])/xdiff

for k in range(1,gridx-1):
    cent_diff[k] = (func_c[k+1] - func_c[k-1])/(2*xdiff)

fig,ax = plt.subplots(1,3,figsize=(12,5),tight_layout=True)
fig.suptitle('Forward, Backward, Centered finite differences against exact derivative')

for axx in ax:
    axx.set_xlim(xval[0],xval[-1])
    axx.set_xlabel('$x$')
    axx.set_ylabel("f'")

ax[0].plot(xval,deriv)
ax[0].plot(xval_c[0:gridx-1],for_diff[0:gridx-1],'go')

ax[1].plot(xval,deriv)
ax[1].plot(xval_c[1:gridx],back_diff[1:gridx],'ro')

ax[2].plot(xval,deriv)
ax[2].plot(xval_c[1:gridx-1],cent_diff[1:gridx-1],'co')

plt.show()


