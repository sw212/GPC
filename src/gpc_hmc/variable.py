from . import transforms
import jax.numpy as jnp

class Variable:
    def __init__(self, value, prior=None, transform=None):
        self.value = jnp.atleast_1d(value)
        self.prior = prior
        if transform is None:
            self.transform = transforms.Identity()
        else:
            self.transform = transform
        self.transformed_value = self.transform.inverse(self.value)
    
    def log_prior_pdf(self):
        return jnp.sum(self.prior.log_pdf(self.value))