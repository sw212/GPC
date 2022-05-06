# # https://nsls-ii.github.io/scientific-python-cookiecutter/preliminaries.html REDO
import pytest
from numpy.testing import assert_allclose
from gpc_hmc import HMC
from jax import numpy as jnp
import jax

class ModelFlat:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def variables(self):
        return {
            'x': self.x,
            'y': self.y,
        }

    def __call__(self, vars):
        return 1.0

class ModelQuadratic:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def variables(self):
        return {
            'x': self.x,
            'y': self.y,
        }

    def __call__(self, vars):
        x = vars[0]
        y = vars[1]
        return x**2 + y**2

# ------------------------- #
# Quadratic Potential Tests #
# ------------------------- #

@pytest.fixture(scope='function')
def model_quadratic():
    model = ModelQuadratic(1.0, 1.0)
    return model

@pytest.fixture()
def hmc_quadratic(model_quadratic):
    hmc_quad = HMC.HMC(model_quadratic)
    return hmc_quad

def test_radial_leap(hmc_quadratic):
    """
    leapfrog radially with radially symmetric potential
    particle 'bearing' or 'angle' should remain unchanged
    """
    hmc_quadratic.p['x'] = 0.5
    hmc_quadratic.p['y'] = 0.5
    init_x, init_y = hmc_quadratic.vars['x'], hmc_quadratic.vars['y']
    init_angle = jnp.arctan2(init_y, init_x)
    q_init = jax.tree_util.tree_leaves(hmc_quadratic.vars)
    p_init = jax.tree_util.tree_leaves(hmc_quadratic.p)

    q_final, p_final = hmc_quadratic.leap(q_init, p_init, 0.01, 50)

    final_x, final_y = q_final[0], q_final[1]
    final_angle = jnp.arctan2(final_y, final_x)
    assert_allclose(init_angle, final_angle, rtol=1e-5)

def test_circular_leap(hmc_quadratic):
    """
    leapfrog perpendicular to force F with F = v^2 / r
    radial distance r should remain unchanged
    """
    hmc_quadratic.vars['x'] = 1.0
    hmc_quadratic.vars['y'] = 0.0
    hmc_quadratic.p['x'] = 0
    hmc_quadratic.p['y'] = jnp.sqrt(2.0)
    init_height = 1.0
    q_init = jax.tree_util.tree_leaves(hmc_quadratic.vars)
    p_init = jax.tree_util.tree_leaves(hmc_quadratic.p)
    
    q_final, p_final = hmc_quadratic.leap(q_init, p_init, 0.001, 50)
    final_height = q_final[0]**2 + q_final[1]**2

    assert_allclose(init_height, final_height, rtol=1e-3)

def test_involution_leap(hmc_quadratic):
    """
    transisiton kernel is an involution
    application of kernel twice should recover initial state
    """
    hmc_quadratic.vars['x'] = 3.2
    hmc_quadratic.vars['y'] = 0.7
    hmc_quadratic.p['x'] = 2.31
    hmc_quadratic.p['y'] = jnp.pi

    q_init = jax.tree_util.tree_leaves(hmc_quadratic.vars)
    p_init = jax.tree_util.tree_leaves(hmc_quadratic.p)
    
    q_middle, p_middle = hmc_quadratic.leap(q_init, p_init, 0.01, 20)

    q_final, p_final = hmc_quadratic.leap(q_middle, p_middle, 0.01, 20)

    assert_allclose(q_init, q_final, atol=1e-3)
    assert_allclose(p_init, p_final, atol=1e-3)

# -------------------- #
# Flat Potential Tests #
# -------------------- #
def test_flat_leap():
    model_flat = ModelFlat(0.0, 0.0)
    hmc_flat = HMC.HMC(model_flat)
    hmc_flat.p['x'] = -1.0
    hmc_flat.p['y'] = 2.0

    num_steps = 10
    q_init = jax.tree_util.tree_leaves(hmc_flat.vars)
    p_init = jax.tree_util.tree_leaves(hmc_flat.p)
    
    q_final, p_final = hmc_flat.leap(q_init, p_init, 1.0, num_steps)

    final_x, final_y = q_final[0], q_final[1]

    expected_x = hmc_flat.p['x'] * num_steps
    expected_y = hmc_flat.p['y'] * num_steps

    assert_allclose(expected_x, final_x, atol=1e-5)
    assert_allclose(expected_y, final_y, atol=1e-5)

def test_stationary_flat_leap():
    model_flat = ModelFlat(1.5, 2.5)
    hmc_flat = HMC.HMC(model_flat)

    init_x, init_y = hmc_flat.vars['x'], hmc_flat.vars['y']
    q_init = jax.tree_util.tree_leaves(hmc_flat.vars)
    p_init = jax.tree_util.tree_leaves(hmc_flat.p)

    q_final, p_final = hmc_flat.leap(q_init, p_init, 1.0, 10)

    final_x, final_y = q_final[0], q_final[1]

    assert_allclose(init_x, final_x, atol=1e-5)
    assert_allclose(init_y, final_y, atol=1e-5)


def test_hmc_normal():
    class Model:
        def __init__(self, x,):
            self.x = jnp.array(x)

        def variables(self):
            return {'x': self.x}

    def neg_log_normal(mu, sigma):
        def logp(x):
            return 0.5 * (jnp.log(2 * jnp.pi * sigma * sigma) + ((x[0] - mu) / sigma) ** 2)
        return logp
    
    potential = neg_log_normal(0.0, 1.0)

    model = Model(0.0)
    hmc = HMC.HMC(model)

    q_init = jax.tree_util.tree_leaves(hmc.vars)
    p_init = jax.tree_util.tree_leaves(hmc.p)


