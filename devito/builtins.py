"""
Built-in Operators provided by Devito.
"""

from sympy import Abs, Pow
import numpy as np

import devito as dv

__all__ = ['assign', 'smooth', 'gaussian_smooth', 'norm', 'sumall', 'inner',
           'mmin', 'mmax']


def assign(f, v=0):
    """
    Assign a value to a Function.

    Parameters
    ----------
    f : Function
        The left-hand side of the assignment.
    v : scalar, optional
        The right-hand side of the assignment.
    """
    dv.Operator(dv.Eq(f, v), name='assign')()


def smooth(f, g, axis=None):
    """
    Smooth a Function through simple moving average.

    Parameters
    ----------
    f : Function
        The left-hand side of the smoothing kernel, that is the smoothed Function.
    g : Function
        The right-hand side of the smoothing kernel, that is the Function being smoothed.
    axis : Dimension or list of Dimensions, optional
        The Dimension along which the smoothing operation is performed. Defaults
        to ``f``'s innermost Dimension.

    Notes
    -----
    More info about simple moving average available at: ::

        https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
    """
    if g.is_Constant:
        # Return a scaled version of the input if it's a Constant
        f.data[:] = .9 * g.data
    else:
        if axis is None:
            axis = g.dimensions[-1]
        dv.Operator(dv.Eq(f, g.avg(dims=axis)), name='smoother')()


def gaussian_smooth(f, sigma=1, _order=4, mode='reflect'):
    """
    Gaussian smooth function.
    """

    class DataDomain(dv.SubDomain):

        name = 'datadomain'

        def __init__(self, lw):
            super(DataDomain, self).__init__()
            self.lw = lw

        def define(self, dimensions):
            return {d: ('middle', self.lw, self.lw) for d in dimensions}

    def fill_data(g, f, dim, lw, mode):

        sl = slice(lw, -lw, 1)
        indices = ()
        for _ in range(len(f.grid.dimensions)):
            indices += (sl, )
        g.data[indices] = f.data
        if mode == 'constant':
            indfl = ()
            indgl = ()
            indfr = ()
            indgr = ()
            for d in f.grid.dimensions:
                if d == dim:
                    indfl += (0, )
                    indgl += (slice(0, lw, 1), )
                    indfr += (-1, )
                    indgr += (slice(-lw, -1, 1), )
                else:
                    indfl += (slice(0, -1, 1), )
                    indgl += (slice(lw, -lw, 1), )
                    indfr += (slice(0, -1, 1), )
                    indgr += (slice(lw, -lw, 1), )
            g.data[indgl] = f.data[indfl]
            g.data[indgr] = f.data[indfr]
        elif mode == 'reflect':
            indfl = ()
            indgl = ()
            indfr = ()
            indgr = ()
            for d in f.grid.dimensions:
                if d == dim:
                    indfl += (slice(lw-1, None, -1), )
                    indgl += (slice(0, lw, 1), )
                    indfr += (slice(-1, -(lw+1), -1), )
                    indgr += (slice(-lw, None, 1), )
                else:
                    indfl += (slice(0, None, 1), )
                    indgl += (slice(lw, -lw, 1), )
                    indfr += (slice(0, None, 1), )
                    indgr += (slice(lw, -lw, 1), )
            g.data[indgl] = f.data[indfl]
            g.data[indgr] = f.data[indfr]
        else:
            raise ValueError("Mode not available")

        return

    def subfloor(f, g):
        sl = slice(lw, -lw, 1)
        indices = ()
        for _ in range(len(f.grid.dimensions)):
            indices += (sl, )
        f.data[:] = np.floor(g.data[indices])

    lw = int(_order*sigma + 0.5)
    dims = f.grid.dimensions

    # Create the padded grid:
    datadomain = DataDomain(lw)
    shape_padded = np.array(f.grid.shape) + 2*lw
    grid = dv.Grid(shape=shape_padded, subdomains=datadomain)

    f_c = dv.Function(name='f_c', grid=grid, space_order=2*lw,
                      coefficients='symbolic')
    f_o = dv.Function(name='f_o', grid=grid, space_order=2*lw,
                      coefficients='symbolic')

    weights = np.exp(-0.5/sigma**2*(np.linspace(-lw, lw, 2*lw+1))**2)
    weights = weights/weights.sum()

    for d in dims:

        fill_data(f_c, f, d, lw, mode)

        rhs = dv.generic_derivative(f_c, d, 2*lw, 1)
        coeffs = dv.Coefficient(1, f_c, d, weights)

        expr = dv.Eq(f_o, rhs, coefficients=dv.Substitutions(coeffs),
                     subdomain=grid.subdomains['datadomain'])
        op = dv.Operator(expr)
        op.apply()
        subfloor(f, f_o)

    return f


# Reduction-inducing builtins

class MPIReduction(object):
    """
    A context manager to build MPI-aware reduction Operators.
    """

    def __init__(self, *functions, op=dv.mpi.MPI.SUM):
        grids = {f.grid for f in functions}
        if len(grids) == 0:
            self.grid = None
        elif len(grids) == 1:
            self.grid = grids.pop()
        else:
            raise ValueError("Multiple Grids found")
        dtype = {f.dtype for f in functions}
        if len(dtype) == 1:
            self.dtype = dtype.pop()
        else:
            raise ValueError("Illegal mixed data types")
        self.v = None
        self.op = op

    def __enter__(self):
        i = dv.Dimension(name='i',)
        self.n = dv.Function(name='n', shape=(1,), dimensions=(i,),
                             grid=self.grid, dtype=self.dtype)
        self.n.data[0] = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.grid is None or not dv.configuration['mpi']:
            assert self.n.data.size == 1
            self.v = self.n.data[0]
        else:
            comm = self.grid.distributor.comm
            self.v = comm.allreduce(np.asarray(self.n.data), self.op)[0]


def norm(f, order=2):
    """
    Compute the norm of a Function.

    Parameters
    ----------
    f : Function
        Input Function.
    order : int, optional
        The order of the norm. Defaults to 2.
    """
    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    p, eqns = f.guard() if f.is_SparseFunction else (f, [])

    with MPIReduction(f) as mr:
        op = dv.Operator(eqns + [dv.Inc(mr.n[0], Abs(Pow(p, order)))],
                         name='norm%d' % order)
        op.apply(**kwargs)

    v = Pow(mr.v, 1/order)

    return np.float(v)


def sumall(f):
    """
    Compute the sum of all Function data.

    Parameters
    ----------
    f : Function
        Input Function.
    """
    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    p, eqns = f.guard() if f.is_SparseFunction else (f, [])

    with MPIReduction(f) as mr:
        op = dv.Operator(eqns + [dv.Inc(mr.n[0], p)], name='sum')
        op.apply(**kwargs)

    return np.float(mr.v)


def inner(f, g):
    """
    Inner product of two Functions.

    Parameters
    ----------
    f : Function
        First input operand
    g : Function
        Second input operand

    Raises
    ------
    ValueError
        If the two input Functions are defined over different grids, or have
        different dimensionality, or their dimension-wise sizes don't match.
        If in input are two SparseFunctions and their coordinates don't match,
        the exception is raised.

    Notes
    -----
    The inner product is the sum of all dimension-wise products. For 1D Functions,
    the inner product corresponds to the dot product.
    """
    # Input check
    if f.is_TimeFunction and f._time_buffering != g._time_buffering:
        raise ValueError("Cannot compute `inner` between save/nosave TimeFunctions")
    if f.shape != g.shape:
        raise ValueError("`f` and `g` must have same shape")
    if f._data is None or g._data is None:
        raise ValueError("Uninitialized input")
    if f.is_SparseFunction and not np.all(f.coordinates_data == g.coordinates_data):
        raise ValueError("Non-matching coordinates")

    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    rhs, eqns = f.guard(f*g) if f.is_SparseFunction else (f*g, [])

    with MPIReduction(f, g) as mr:
        op = dv.Operator(eqns + [dv.Inc(mr.n[0], rhs)], name='inner')
        op.apply(**kwargs)

    return np.float(mr.v)


def mmin(f):
    """
    Retrieve the minimum.

    Parameters
    ----------
    f : array_like or Function
        Input operand.
    """
    if not isinstance(f, dv.Function):
        return np.min(f)

    with MPIReduction(f, op=dv.mpi.MPI.MIN) as mr:
        mr.n.data[0] = np.min(f.data_ro_domain).item()
    return mr.v.item()


def mmax(f):
    """
    Retrieve the maximum.

    Parameters
    ----------
    f : array_like or Function
        Input operand.
    """
    if not isinstance(f, dv.Function):
        return np.max(f)

    with MPIReduction(f, op=dv.mpi.MPI.MAX) as mr:
        mr.n.data[0] = np.max(f.data_ro_domain).item()
    return mr.v.item()
