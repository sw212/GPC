import jax.numpy as jnp
from jax.lax import lgamma


class Gamma:
    """
    Implements the logarithm of the Gamma PDF w.r.t. shape (a), rate (b) parameterisation.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def log_pdf(self, x):
        return (self.a * jnp.log(self.b)) + ((self.a - 1) * jnp.log(x)) - (self.b * x) - lgamma(self.a)

    def log_pdf_grad(self, x):
        return jnp.divide(self.a - 1, x) - self.b

class Normal:
    """
    Implements the logarithm of the Normal PDF w.r.t. initialised mean and variance.
    """
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def log_pdf(self, x):
        constant = jnp.log(jnp.sqrt(self.var)) + .5*jnp.log(2.0 * jnp.pi) 
        log_exp = -.5 * jnp.divide(jnp.square(x - self.mean), self.var)
        return log_exp - constant