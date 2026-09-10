from jax import random
from jax import numpy as jnp

import jax

import equinox as eqx
from diffrax import *
import lineax

class RNN(eqx.Module):
    """A simple Recurrent Neural Network using Equinox."""
    W: jax.Array
    B: list[jax.Array]
    D: jax.Array #= eqx.field(static=True)
    activation: callable

    dim: int
    input_dims: list[int]
    output_dim: int

    gain: float

    def __init__(self, key, dim, input_dims, output_dim, activation=jax.nn.tanh, gain=1.0):

        self.dim = dim
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.activation = activation
        self.gain = gain

        key, key_W, key_B, key_D = random.split(key, 4)
        self.W = gain*random.normal(key_W, (self.dim, self.dim))/jnp.sqrt(self.dim)

        key_Bs = random.split(key_B, len(self.input_dims))
        self.B = [random.normal(k, (self.dim, i))/jnp.sqrt(i) for i, k in zip(input_dims, key_Bs)]

        self.D =  random.normal(key_D, (self.output_dim, self.dim))/jnp.sqrt(self.dim)

    def F(self, x):

        return self.W @ self.activation(x) - x

    def G(self, i):

        return self.B[i]

    def H(self, sigma):

        return lineax.DiagonalLinearOperator(jnp.full(self.dim, sigma))

    def f(self):

        return lambda t, x, args: self.F(x)

    def g_i(self, i):

        return lambda t, x, args: self.G(i)

    def h_g_i(self, i, sigma):

        return lambda t, x, args: self.G(i)*sigma

    def h_sigma(self, sigma):

        return lambda t, x, args: self.H(sigma)

    def d(self, x):

        return self.activation(x) @ self.D.T

    def x0(self):

        return jnp.zeros(self.dim)


if __name__ == "__main__":

    input_dims = [5, 7]

    rnn = RNN(jax.random.PRNGKey(0), 10, input_dims)

    bms = [VirtualBrownianTree(shape=(i,), t0=0.0, t1=1.0, tol=10**-3, key=jax.random.PRNGKey(1)) for i in input_dims]

    #bms = [LinearInterpolation(jnp.linspace(0, 1.0, 11), random.uniform(random.key(9), (11, i))) for i in input_dims]

    terms = [ODETerm(rnn.f())]
    terms = terms + [ControlTerm(rnn.g_i(i), b) for i, b in enumerate(bms)]
    terms = MultiTerm(*terms)

    sol = diffeqsolve(terms, Heun(), 0.0, 1.0, 0.01, rnn.x0(), saveat=SaveAt(ts=jnp.linspace(0.0, 1.0, 101)))

    import matplotlib.pyplot as plt
    plt.plot(sol.ts, sol.ys[:, :2])

    plt.show()