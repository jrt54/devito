import numpy as np
from sympy import cos, sin

from devito import Function, TimeFunction, memoized_meth, Inc, Eq, DefaultDimension, ConditionalDimension, solve, Operator
from examples.seismic import PointSource, Receiver
from examples.seismic.acoustic.operators import (
    ForwardOperator, AdjointOperator, GradientOperator, BornOperator
)
from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver


class AcousticWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    :param model: Physical model with domain parameters
    :param source: Sparse point symbol providing the injected wave
    :param receiver: Sparse point symbol describing an array of receivers
    :param time_order: Order of the time-stepping scheme (default: 2, choices: 2,4)
                       time_order=4 will not implement a 4th order FD discretization
                       of the time-derivative as it is unstable. It implements instead
                       a 4th order accurate wave-equation with only second order
                       time derivative. Full derivation and explanation of the 4th order
                       in time can be found at:
                       http://www.hl107.math.msstate.edu/pdfs/rein/HighANM_final.pdf
    :param space_order: Order of the spatial stencil discretisation (default: 4)

    Note: space_order must always be greater than time_order
    """
    def __init__(self, model, source, receiver, kernel='OT2', space_order=2, **kwargs):
        self.model = model
        self.source = source
        self.receiver = receiver

        self.space_order = space_order
        self.kernel = kernel

        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        self.dt = self.model.critical_dt
        if self.kernel == 'OT4':
            self.dt *= 1.73

        # Cache compiler options
        self._kwargs = kwargs

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, source=self.source,
                               receiver=self.receiver, kernel=self.kernel,
                               space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, source=self.source,
                               receiver=self.receiver, kernel=self.kernel,
                               space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True):
        """Cached operator for gradient runs"""
        return GradientOperator(self.model, save=save, source=self.source,
                                receiver=self.receiver, kernel=self.kernel,
                                space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_born(self):
        """Cached operator for born runs"""
        return BornOperator(self.model, save=None, source=self.source,
                            receiver=self.receiver, kernel=self.kernel,
                            space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec=None, u=None, m=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        :param src: Symbol with time series data for the injected source term
        :param rec: Symbol to store interpolated receiver data
        :param u: (Optional) Symbol to store the computed wavefield
        :param m: (Optional) Symbol for the time-constant square slowness
        :param save: Option to store the entire (unrolled) wavefield

        :returns: Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.source
        # Create a new receiver object to store the result
        rec = rec or Receiver(name='rec', grid=self.model.grid,
                              time_range=self.receiver.time_range,
                              coordinates=self.receiver.coordinates.data)

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.source.nt if save else None,
                              time_order=2, space_order=self.space_order)

        # Pick m from model unless explicitly provided
        m = m or self.model.m

        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec=rec, u=u, m=m,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, summary

    def adjoint(self, rec, srca=None, v=None, m=None, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        :param rec: Symbol with stored receiver data. Please note that
                    these act as the source term in the adjoint run.
        :param srca: Symbol to store the resulting data for the
                     interpolated at the original source location.
        :param v: (Optional) Symbol to store the computed wavefield
        :param m: (Optional) Symbol for the time-constant square slowness

        :returns: Adjoint source, wavefield and performance summary
        """
        # Create a new adjoint source and receiver symbol
        srca = srca or PointSource(name='srca', grid=self.model.grid,
                                   time_range=self.source.time_range,
                                   coordinates=self.source.coordinates.data)

        # Create the adjoint wavefield if not provided
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick m from model unless explicitly provided
        m = m or self.model.m

        # Execute operator and return wavefield and receiver data
        summary = self.op_adj().apply(srca=srca, rec=rec, v=v, m=m,
                                      dt=kwargs.pop('dt', self.dt), **kwargs)
        return srca, v, summary

    def gradient(self, rec, u, v=None, grad=None, m=None, checkpointing=False, **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        :param recin: Receiver data as a numpy array
        :param u: Symbol for full wavefield `u` (created with save=True)
        :param v: (Optional) Symbol to store the computed wavefield
        :param grad: (Optional) Symbol to store the gradient field

        :returns: Gradient field and performance summary
        """
        dt = kwargs.pop('dt', self.dt)
        # Gradient symbol
        grad = grad or Function(name='grad', grid=self.model.grid)

        # Create the forward wavefield
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick m from model unless explicitly provided
        m = m or self.model.m

        if checkpointing:
            u = TimeFunction(name='u', grid=self.model.grid,
                             time_order=2, space_order=self.space_order)
            cp = DevitoCheckpoint([u])
            n_checkpoints = None
            wrap_fw = CheckpointOperator(self.op_fwd(save=False), src=self.source,
                                         u=u, m=m, dt=dt)
            wrap_rev = CheckpointOperator(self.op_grad(save=False), u=u, v=v,
                                          m=m, rec=rec, dt=dt, grad=grad)

            # Run forward
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2)
            wrp.apply_forward()
            summary = wrp.apply_reverse()
        else:
            summary = self.op_grad().apply(rec=rec, grad=grad, v=v, u=u, m=m,
                                           dt=dt, **kwargs)
        return grad, summary

    def born(self, dmin, src=None, rec=None, u=None, U=None, m=None, **kwargs):
        """
        Linearized Born modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        :param src: Symbol with time series data for the injected source term
        :param rec: Symbol to store interpolated receiver data
        :param u: (Optional) Symbol to store the computed wavefield
        :param U: (Optional) Symbol to store the computed wavefield
        :param m: (Optional) Symbol for the time-constant square slowness
        """
        # Source term is read-only, so re-use the default
        src = src or self.source
        # Create a new receiver object to store the result
        rec = rec or Receiver(name='rec', grid=self.model.grid,
                              time_range=self.receiver.time_range,
                              coordinates=self.receiver.coordinates.data)

        # Create the forward wavefields u and U if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        U = U or TimeFunction(name='U', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick m from model unless explicitly provided
        m = m or self.model.m

        # Execute operator and return wavefield and receiver data
        summary = self.op_born().apply(dm=dmin, u=u, U=U, src=src, rec=rec,
                                       m=m, dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, U, summary

    def forward_freq_modeling(self, freq=None, factor=None, **kwargs):
        # Forward modeling with on-the-fly DFT of forward wavefields
        # Parameters
        nt = self.source.nt
        dt = self.model.critical_dt
        m, damp = self.model.m, self.model.damp
        nfreq = len(freq)
        freq_dim = DefaultDimension(name='freq_dim', default_value=nfreq)
        time = self.model.grid.time_dim
        if factor is None:
            factor = int(1 / (dt*4*np.max(freq)))
        if factor==1:
            tsave = time
        else:
            tsave = ConditionalDimension(name='tsave', parent=self.model.grid.time_dim, factor=factor)

        # Create wavefields
        u = TimeFunction(name='u', grid=self.model.grid, time_order=2, space_order=self.space_order)
        f = Function(name='f', dimensions=(freq_dim,), shape=(nfreq,))
        f.data[:] = freq[:]
        ufr = Function(name='ufr', dimensions=(freq_dim,) + u.indices[1:], shape=(nfreq,) + u.shape[1:])
        ufi = Function(name='ufi', dimensions=(freq_dim,) + u.indices[1:], shape=(nfreq,) + u.shape[1:])

        # Set up PDE and rearrange
        eqn = m * u.dt2 - u.laplace + damp * u.dt
        stencil = solve(eqn, u.forward)
        expression = [Eq(u.forward, stencil)]

        # Source symbol with input wavelet
        src = PointSource(name='src', grid=self.model.grid,
                          time_range=self.source.time_range,
                          coordinates=self.source.coordinates.data)
        src_term = src.inject(field=u.forward, offset=self.model.nbpml, expr=src * dt**2 / m)

        # Data is sampled at receiver locations
        rec = Receiver(name='rec', grid=self.model.grid,
                       time_range=self.receiver.time_range,
                       coordinates=self.receiver.coordinates.data)
        rec_term = rec.interpolate(expr=u, offset=self.model.nbpml)

        # Create operator and run
        expression += src_term + rec_term
        expression += [Inc(ufr, factor*u*cos(2*np.pi*f*tsave*factor*dt))]
        expression += [Inc(ufi, - factor*u*sin(2*np.pi*f*tsave*factor*dt))]
        subs = self.model.spacing_map
        subs[u.grid.time_dim.spacing] = self.model.critical_dt
        op = Operator(expression, subs=subs, dse='advanced', dle='advanced')
        op(**kwargs)

        return rec, ufr, ufi


    def adjoint_freq_born(self, recin, freq, factor, ufr, ufi, **kwargs):

        # Parameters
        nt = self.source.nt
        dt = self.model.critical_dt
        m, damp = self.model.m, self.model.damp
        nfreq = ufr.shape[0]
        time = self.model.grid.time_dim
        if factor is None:
            factor = int(1 / (dt*4*np.max(freq)))
        if factor==1:
            tsave = time
        else:
            tsave = ConditionalDimension(name='tsave', parent=self.model.grid.time_dim, factor=factor)
        dtf = factor * dt
        ntf = factor / nt
        print("DFT subsampling factor: ", factor)

        # Create the forward and adjoint wavefield
        v = TimeFunction(name='v', grid=self.model.grid, time_order=2, space_order=self.space_order)
        f = Function(name='f', dimensions=(ufr.indices[0],), shape=(nfreq,))
        f.data[:] = freq[:]
        gradient = Function(name="grad", grid=self.model.grid)

        # Set up PDE and rearrange
        eqn = m * v.dt2 - v.laplace - damp * v.dt
        stencil = solve(eqn, v.backward)
        expression = [Eq(v.backward, stencil)]

        # Data at receiver locations as adjoint source
        rec = Receiver(name='rec', grid=self.model.grid,
                       time_range=self.receiver.time_range,
                       coordinates=self.receiver.coordinates.data)
        adj_src = rec.inject(field=v.backward, offset=self.model.nbpml, expr=rec * dt**2 / m)

        # Gradient update
        gradient_update = [Eq(gradient, gradient + (2*np.pi*f)**2/nt*(ufr*cos(2*np.pi*f*tsave*dtf) - ufi*sin(2*np.pi*f*tsave*dtf))*v)]

        # Create operator and run
        expression += adj_src + gradient_update
        subs = self.model.spacing_map
        subs[v.grid.time_dim.spacing] = dt
        op = Operator(expression, subs=subs, dse='advanced', dle='advanced')

        op(rec=recin, **kwargs)
        return gradient