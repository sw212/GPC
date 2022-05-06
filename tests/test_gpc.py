import pytest
from numpy.testing import assert_allclose
from gpc_hmc.transforms import LogExp
from gpc_hmc.variable import Variable
from gpc_hmc.prior import Gamma
from gpc_hmc.kernel import RBF
from gpc_hmc.GPC import GPClassification
from jax import numpy as jnp
from jax import scipy as jsp
import jax
import numpy as np

def test_model_log_lh_1():
    X = np.linspace(-5, 5, 10)[:, np.newaxis]
    y = np.ones_like(X)
    variance = Variable(1.0, Gamma(1.0, 1.0), LogExp)
    lengthscale = Variable(1.0, Gamma(1.0, 1.0), LogExp)

    rbf = RBF(variance, lengthscale)
    model = GPClassification(X, y, rbf)
    maps = jax.tree_util.tree_leaves(model.mappings())
    vars = jax.tree_util.tree_leaves(model.variables())

    vars = jax.tree_util.tree_map(lambda f, x: f(x), maps, vars)
    log_lh = model._log_lh(vars)

    expected_log_lh = jnp.log(0.5) * 10

    assert_allclose(log_lh, expected_log_lh)


def test_model_log_prior_1():
    X = np.linspace(-5, 5, 10)[:, np.newaxis]
    y = np.ones_like(X)
    variance = Variable(1.0, Gamma(1.0, 1.0), LogExp)
    lengthscale = Variable(1.0, Gamma(1.0, 1.0), LogExp)

    rbf = RBF(variance, lengthscale)
    model = GPClassification(X, y, rbf)
    maps = jax.tree_util.tree_leaves(model.mappings())
    vars = jax.tree_util.tree_leaves(model.variables())

    vars = jax.tree_util.tree_map(lambda f, x: f(x), maps, vars)
    log_prior = model._log_prior(vars)

    l_gamma = jsp.stats.gamma.logpdf(x=1.0, a=1.0)
    l_normal = jsp.stats.norm.logpdf(x=0.0)
    expected_log_prior = (2 * l_gamma) + (10 * l_normal)

    assert_allclose(log_prior, expected_log_prior)

