from copy import deepcopy
import jax
from jax import numpy as jnp

class HMC:
    def __init__(self, model) -> None:
        self.model = model
        self.key = jax.random.PRNGKey(42)
        self.subkey = jax.random.split(self.key)
        self.vars = model.variables()
        self.p = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), model.variables())
        self.current_state = None
        self.proposed_state = None    

    def hamiltonian(self, q, p):
        '''
        Computes the Hamiltonian of current state i.e. "H = K.E. - P.E."
        up to normalistation assuming a unit mass matrix.
        '''
        add_squares = lambda x: jnp.sum(jnp.square(x))

        sum_squares = jax.tree_util.tree_map(add_squares, p)
        sum_squares = jax.tree_util.tree_reduce(jnp.add, sum_squares)
        return 0.5 * sum_squares + self.model(q)

    def leap(self, q, p, step_size, leaps):
        '''
        Performs basic leapfrog of state using standard Stormer-Verlet integrator.
        q - position
        p - momentum
        potential - ...
        step_size : int - size of integrator step
        leaps : int - num of discrete leapfrog steps
        '''
        V, dV_dq = jax.value_and_grad(self.model)(q)
        p = jax.tree_util.tree_map(lambda p, g: p - 0.5*step_size*g, p, dV_dq)

        for leap in range(leaps):
            q = jax.tree_util.tree_map(lambda v, p: v + step_size*p, q, p)
            V, dV_dq = jax.value_and_grad(self.model)(q)
            p = jax.tree_util.tree_map(lambda p, g: p - step_size*g, p, dV_dq)

        # p is flipped to make kernel an involution
        p = jax.tree_util.tree_map(lambda p, g: -(p + 0.5*step_size*g), p, dV_dq)

        return q, p
        
    def sample(self, num_samples, q, p, step_size, leaps):
        '''
        Basic HMC sampling of state with standard MH acceptence correction.
        '''
        num_components = len(q)
        samples = [None] * num_samples
        samples[0] = q

        for iter in range(num_samples - 1):
            self.key, *subkeys = jax.random.split(self.key, num=1 + num_components)
            p = jax.tree_util.tree_map(lambda arr, key: jax.random.normal(key, arr.shape), q, subkeys)
            old_p = deepcopy(p)
            old_q = deepcopy(q)
            old_H = self.hamiltonian(q, p)

            q, p = self.leap(q, p, step_size, leaps)

            new_H = self.hamiltonian(q, p)

            self.key, subkey = jax.random.split(self.key)
            if jnp.log(jax.random.uniform(subkey)) > old_H - new_H:
                p = old_p
                q = old_q

            samples[iter] = deepcopy(q)
        
        return samples

    def sample_jitted(self, num_samples, q, p, step_size, leaps):
        '''
        Basic HMC sampling of state with standard MH acceptence correction.
        JIT comples leapfrog and Hamiltonian calculation
        '''
        H = jax.jit(self.hamiltonian)
        leaper = jax.jit(self.leap, static_argnames=['leaps'])
        num_components = len(q)
        samples = [None] * num_samples
        samples[0] = q

        for iter in range(num_samples - 1):
            self.key, *subkeys = jax.random.split(self.key, num=1 + num_components)
            p = jax.tree_util.tree_map(lambda arr, key: jax.random.normal(key, arr.shape), q, subkeys)
            old_p = deepcopy(p)
            old_q = deepcopy(q)
            old_H = H(q, p)

            q, p = leaper(q, p, step_size, leaps)

            new_H = H(q, p)

            self.key, subkey = jax.random.split(self.key)
            if jnp.log(jax.random.uniform(subkey)) > old_H - new_H:
                p = old_p
                q = old_q

            samples[iter + 1] = deepcopy(q)
        
        return samples