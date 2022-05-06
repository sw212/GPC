import jax
from jax import numpy as jnp
from jax.scipy.special import erf


def inv_probit(x):
    """
    CDF of a N(0, 1) random variable.
    """
    return 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))

def rng_keys_tree(key, tree):
    """
    Generate consumable PRNG keys conforming to pytree structure.
    """
    structure = jax.tree_util.tree_structure(tree)
    keys = jax.random.split(key, structure.num_leaves)
    return jax.tree_util.tree_unflatten(structure, keys)


def rng_normals_tree(key, tree):
    """
    Generate N(0,1) random variables conforming to a pytree structure.
    """
    keys = rng_keys_tree(key, tree)
    return jax.tree_util.tree_map(lambda t, k: jax.random.normal(k, t.shape), tree, keys)