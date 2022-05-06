import jax.numpy as jnp
from .variable import Variable

class RBF:
    def __init__(self, variance, lengthscale):
        if not (isinstance(variance, Variable) and isinstance(lengthscale, Variable)): 
            raise TypeError('Hyperparameters must both be Variable instances.')

        self.variance = variance
        self.lengthscale = lengthscale

    def K(self, x1, x2):
        """
        RBF cov matrix between x1 and x2
        x1 : points - array of shape (n, 1)
        x2 : points - array of shape (n, 1)
        """
        exponent = jnp.einsum('ij,ij->i', x1, x1)[:, jnp.newaxis] + jnp.einsum('ij,ij->i', x2, x2)[jnp.newaxis, :] - (2 * (x1 @ jnp.transpose(x2)))
        return self.variance.value * jnp.exp(-exponent / (2 * (self.lengthscale.value ** 2)))
    
    def variables(self):
        vars = {
            'variance': self.variance.transformed_value,
            'lengthscale': self.lengthscale.transformed_value
            }
        return vars
    
    def mappings(self):
        maps = {
            'variance': self.variance.transform.forward,
            'lengthscale': self.lengthscale.transform.forward
        }
        return maps

    def priors(self):
        priors = {
            'variance': self.variance.prior.log_pdf,
            'lengthscale': self.lengthscale.prior.log_pdf
        }
        return priors

    def __call__(self, vars, x1, x2):
        """
        RBF cov matrix between x1 and x2
        x1 : points - array of shape (n, 1)
        x2 : points - array of shape (n, 1)
        variance : rbf kernel variance (float > 0) 
        lengthscale : rbf kernel lengthscale (float > 0)
        """
        # order is alphabetical ascending i.e. 1st lengthscale, 2nd variance
        lengthscale, variance = vars
        exponent = jnp.einsum('ij,ij->i', x1, x1)[:, jnp.newaxis] + jnp.einsum('ij,ij->i', x2, x2)[jnp.newaxis, :] - (2 * (x1 @ jnp.transpose(x2)))

        return variance * jnp.exp(-exponent / (2.0 * (lengthscale ** 2)))
        