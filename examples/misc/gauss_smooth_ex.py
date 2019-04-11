import numpy as np
import sympy

from devito import Grid, Function
from devito.builtins import gaussian_smooth

from scipy.ndimage import gaussian_filter, correlate1d
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib import cm

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
f = Function(name='f', grid=grid, dtype=np.int32)
f.data[:] = ascent

f = gaussian_smooth(f, sigma=5)

"""
Plots:
"""

diff = result-f.data[:]

fig = plt.figure()
plt.gray()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.imshow(ascent)
ax2.imshow(diff)
ax3.imshow(result)
ax4.imshow(f.data[:])
plt.show()

#########################################################
from IPython import embed; embed()
#########################################################
