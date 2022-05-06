import jax.numpy as jnp

class Identity:
    '''
    Identity map i.e. f(x) = x.
    '''
    @staticmethod
    def forward(x):
        return x
    @staticmethod
    def inverse(x):
        return x

class LogExp:
    """
    Bijection between reals and positive reals.
    """
    @staticmethod
    def forward(x):
        return jnp.log(1.0 + jnp.exp(x))

    @staticmethod
    def inverse(y):
        return jnp.log(jnp.expm1(y))

    @staticmethod
    def log_jacobian(x):
        return -1.0 * jnp.log(1.0 + jnp.exp(-x))