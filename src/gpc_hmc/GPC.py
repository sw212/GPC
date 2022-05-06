from .variable import Variable
from . import prior
from .likelihood import Bernoulli
from . import transforms
from .util import inv_probit
import jax.numpy as jnp
import jax

class GPClassification:
    def __init__(self, X, y, kernel, likelihood = Bernoulli(inv_link=inv_probit)):
        '''
        X - domain values
        y - class values
        z - GP function values
        '''
        self.X = X
        self.y = y
        self.z = Variable(jnp.zeros_like(X), prior=prior.Normal(0, 1), transform=transforms.Identity)

        self.mean = 0.0
        self.kernel = kernel
        self.likelihood = likelihood


    def variables(self):
        k_vars = self.kernel.variables()
        return {
            "z": self.z.transformed_value,
            "kernel": k_vars
        }
    
    def mappings(self):
        k_maps = self.kernel.mappings()
        return {
            "z": self.z.transform.forward,
            "kernel": k_maps
        }
    
    def priors(self):
        k_priors = self.kernel.priors()
        return {
            "z": self.z.prior.log_pdf,
            "kernel": k_priors
        }

    def __call__(self, vars):
        """
        compute the model log posterior w.r.t the unconstrained values
        each var will be transformed via the 'forward' transformation to the constained space and then used for probability calculations
        """
        maps = jax.tree_util.tree_leaves(self.mappings()) # alphabetical ascending order

        jac_vars = jax.tree_util.tree_map(lambda f, x: jax.vmap(jax.jacobian(f))(x), maps, vars)
        log_jac_vars = jax.tree_util.tree_map(lambda x: jnp.sum(jax.vmap(jnp.log)(x)), jac_vars)
        tot_log_jac_vars = jax.tree_util.tree_reduce(jnp.add, log_jac_vars)

        vars = jax.tree_util.tree_map(lambda f, x: f(x), maps, vars)
        log_lh = self._log_lh(vars)
        log_prior = self._log_prior(vars)

        return log_lh + log_prior + tot_log_jac_vars
    
    def _log_lh(self, vars):
        K = self.kernel(vars[:-1], self.X, self.X)
        L = jnp.linalg.cholesky(K + (jnp.eye(self.X.shape[0])*1e-6))
        f = self.mean + (L @ vars[-1])
        return jnp.sum(self.likelihood.log_prob(f, self.y))
    
    def _log_prior(self, vars):
        priors = jax.tree_util.tree_leaves(self.priors())
        log_prior = jax.tree_util.tree_map(lambda f, x: jnp.sum(jax.vmap(f)(x)), priors, vars)
        log_prior = jax.tree_util.tree_reduce(jnp.add, log_prior)
        return log_prior