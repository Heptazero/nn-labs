import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from typing import Callable

# Written with gemini 2.5 pro

class MLP(eqx.Module):
    """A simple Multi-Layer Perceptron using Equinox."""
    layers: list
    activation: Callable

    def __init__(self, in_size: int, out_size: int, width_size: int, depth: int, activation: Callable = jax.nn.tanh, *, key: jax.random.PRNGKey):
        """
        Initialises the MLP.

        Args:
            in_size: Dimension of the input features.
            out_size: Dimension of the output.
            width_size: Dimension of the hidden layers.
            depth: Number of hidden layers. A depth of 0 means a single linear layer
                   from input to output.
            activation: The activation function (e.g., jax.nn.tanh, jax.nn.relu).
                        Defaults to jax.nn.tanh.
            key: A JAX PRNG key for parameter initialisation.
        """
        keys = jax.random.split(key, depth + 1)
        self.layers = []

        if depth == 0:
            # Network consists of a single linear layer
            self.layers.append(eqx.nn.Linear(in_size, out_size, key=keys[0]))
        else:
            # Input layer (in_size -> width_size)
            self.layers.append(eqx.nn.Linear(in_size, width_size, key=keys[0]))
            # Hidden layers (width_size -> width_size)
            for i in range(depth - 1):
                self.layers.append(eqx.nn.Linear(width_size, width_size, key=keys[i + 1]))
            # Output layer (width_size -> out_size)
            self.layers.append(eqx.nn.Linear(width_size, out_size, key=keys[depth], use_bias=False))

        self.activation = activation

    def __call__(self, x: jax.Array, key=None) -> jax.Array:
        """
        Performs the forward pass of the MLP.

        Args:
            x: The input JAX array. It should have a shape compatible with `in_size`
               (e.g., (batch, in_size) or (in_size,)).

        Returns:
            The output JAX array from the network.
        """
        # Apply layers sequentially
        for i, layer in enumerate(self.layers[:-1]):
            if key is not None:
                key, subkey = jax.random.split(key)
                x = x + random.normal(subkey, x.shape) * 0.5
             # Apply linear layer and activation for all but the last layer
            x = self.activation(layer(x))
        # Apply the final layer without activation
        x = self.layers[-1](x)
        return x

    def get_activity(self, x: jax.Array, layer_index: int) -> jax.Array:
        """
        Gets the activity (output) at a specific layer index.

        Args:
            x: The input JAX array.
            layer_index: The index of the layer whose output activity is needed (0-based).
                         Index 0 corresponds to the output of the first layer (after activation,
                         if applicable). The last index corresponds to the final network output.

        Returns:
            The JAX array representing the activity at the specified layer.

        Raises:
            ValueError: If `layer_index` is out of the valid range [0, len(self.layers) - 1].
        """
        if not (0 <= layer_index < len(self.layers)):
            raise ValueError(f"layer_index must be between 0 and {len(self.layers) - 1}, got {layer_index}")

        _x = x
        for i, layer in enumerate(self.layers):
            # Apply the linear transformation
            _x = layer(_x)
            # Apply activation if it's not the final layer
            if i < len(self.layers) - 1:
                 _x = self.activation(_x)
            # Return the result if we reached the desired layer index
            if i == layer_index:
                return _x
        # Fallback return, though the check above should handle indices correctly.
        # This would return the final output if layer_index was the last valid index.
        return _x