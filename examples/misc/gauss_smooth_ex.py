import numpy as np
import sympy

from devito import Grid, Function
from devito.builtins import gaussian_smooth

from scipy.ndimage import gaussian_filter, correlate1d
from scipy import misc
import matplotlib.pyplot as plt

"""
Python ndimage:
"""

ascent = misc.ascent()
result = gaussian_filter(ascent, sigma=5)

"""
Using Devito:
"""

grid = Grid(shape = ascent.shape, extent = ascent.shape)
x = grid.dimensions
f = Function(name='f', grid=grid)
f.data[:] = ascent

gaussian_smooth(f, sigma=5)

"""
Plots:
"""

fig = plt.figure()
plt.gray()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(ascent)
ax2.imshow(result)
ax3.imshow(f.data[:])
plt.show()

##########################################################
#from IPython import embed; embed()
##########################################################
