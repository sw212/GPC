import jax.numpy as jnp

class Bernoulli:
    def __init__(self, inv_link=None):
        if inv_link is None:
            self.inv_link = lambda x: x
        else:
            self.inv_link = inv_link

    def prob(self, x, k):
        p = self.inv_link(x)
        return jnp.where(jnp.equal(k, 1), p, 1-p)
    
    def log_prob(self, x, k):
        return jnp.log(self.prob(x, k))