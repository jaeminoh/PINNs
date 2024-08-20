import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class MLP(eqx.Module):
    layers: list
    