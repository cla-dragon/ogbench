import functools
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class MLP(nnx.Module):
    """Multi-layer perceptron using flax.nnx with lazy initialization."""
    
    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations=nnx.gelu,
        activate_final: bool = False,
        kernel_init=None,
        layer_norm: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        self.kernel_init = kernel_init or nnx.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')
        self.rngs = rngs
        
        # Initialize as None, will be created on first call
        self.layers = None
        self.initialized = False
    
    def _create_layers(self, input_dim):
        """Create layers when input dimension is known."""
        if self.initialized:
            return
            
        self.layers = []
        current_dim = input_dim
        
        for i, size in enumerate(self.hidden_dims):
            layer = nnx.Linear(
                in_features=current_dim,
                out_features=size,
                kernel_init=self.kernel_init,
                rngs=self.rngs,
            )
            self.layers.append(layer)
            current_dim = size
            
            if (i + 1 < len(self.hidden_dims) or self.activate_final) and self.layer_norm:
                ln = nnx.LayerNorm(num_features=size, rngs=self.rngs)
                self.layers.append(ln)
        
        self.initialized = True
    
    def __call__(self, x):
        if not self.initialized:
            self._create_layers(x.shape[-1])
        
        for i, size in enumerate(self.hidden_dims):
            # Apply linear layer
            layer_idx = i * 2 if self.layer_norm else i
            x = self.layers[layer_idx](x)
            
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    # Apply layer norm
                    ln_idx = layer_idx + 1
                    x = self.layers[ln_idx](x)
        return x


class ResNetBlock(nnx.Module):
    """Individual ResNet block using flax.nnx."""
    
    def __init__(self, num_features: int, rngs: nnx.Rngs = None):
        self.num_features = num_features
        self.rngs = rngs
        
        # Initialize as None, will be created on first call
        self.conv1 = None
        self.conv2 = None
        self.initialized = False
    
    def _create_layers(self, in_features):
        """Create layers when input features are known."""
        if self.initialized:
            return
            
        initializer = nnx.initializers.xavier_uniform()
        
        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            kernel_init=initializer,
            rngs=self.rngs,
        )
        
        self.conv2 = nnx.Conv(
            in_features=self.num_features,
            out_features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            kernel_init=initializer,
            rngs=self.rngs,
        )
        
        self.initialized = True
    
    def __call__(self, x):
        if not self.initialized:
            self._create_layers(x.shape[-1])
            
        block_input = x
        
        x = nnx.relu(x)
        x = self.conv1(x)
        
        x = nnx.relu(x)
        x = self.conv2(x)
        
        # Residual connection
        x = x + block_input
        
        return x


class ResnetStack(nnx.Module):
    """ResNet stack module using flax.nnx."""

    def __init__(
        self, 
        num_features: int, 
        num_blocks: int, 
        max_pooling: bool = True,
        rngs: nnx.Rngs = None,
    ):
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling
        self.rngs = rngs
        
        # Initialize as None, will be created on first call
        self.initial_conv = None
        self.blocks = None
        self.initialized = False
    
    def _create_layers(self, in_features):
        """Create layers when input features are known."""
        if self.initialized:
            return
            
        initializer = nnx.initializers.xavier_uniform()
        
        # Initial convolution
        self.initial_conv = nnx.Conv(
            in_features=in_features,
            out_features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
            rngs=self.rngs,
        )
        
        # ResNet blocks
        self.blocks = []
        for _ in range(self.num_blocks):
            block = ResNetBlock(self.num_features, rngs=self.rngs)
            self.blocks.append(block)
        
        self.initialized = True

    def __call__(self, x):
        if not self.initialized:
            self._create_layers(x.shape[-1])
            
        conv_out = self.initial_conv(x)
        
        if self.max_pooling:
            conv_out = nnx.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )
        
        for block in self.blocks:
            conv_out = block(conv_out)
        
        return conv_out


class ImpalaEncoder(nnx.Module):
    """IMPALA encoder using flax.nnx."""

    def __init__(
        self,
        width: int = 1,
        stack_sizes: tuple = (16, 32, 32),
        num_blocks: int = 2,
        dropout_rate: float = None,
        mlp_hidden_dims: Sequence[int] = (512,),
        layer_norm: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.mlp_hidden_dims = mlp_hidden_dims
        self.layer_norm = layer_norm
        self.rngs = rngs
        
        # Create ResNet stacks
        self.stack_blocks = []
        for i in range(len(stack_sizes)):
            stack = ResnetStack(
                num_features=stack_sizes[i] * width,
                num_blocks=num_blocks,
                rngs=rngs,
            )
            self.stack_blocks.append(stack)
        
        # Dropout layer if needed
        if dropout_rate is not None:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        
        # Layer normalization and MLP will be created on first call
        self.layer_norm_module = None
        self.mlp = None

    def __call__(self, x, train=True, cond_var=None):
        # Normalize input
        x = x.astype(jnp.float32) / 255.0
        
        conv_out = x
        
        # Apply ResNet stacks
        for idx, stack_block in enumerate(self.stack_blocks):
            conv_out = stack_block(conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)
        
        conv_out = nnx.relu(conv_out)
        
        if self.layer_norm:
            if self.layer_norm_module is None:
                self.layer_norm_module = nnx.LayerNorm(num_features=conv_out.shape[-1], rngs=self.rngs)
            conv_out = self.layer_norm_module(conv_out)
        
        # Flatten
        out = conv_out.reshape((*x.shape[:-3], -1))
        
        # Apply MLP
        if self.mlp is None:
            self.mlp = MLP(
                self.mlp_hidden_dims, 
                activate_final=True, 
                layer_norm=self.layer_norm,
                rngs=self.rngs,
            )
        out = self.mlp(out)
        
        return out


class GCEncoder(nnx.Module):
    """Helper module to handle inputs to goal-conditioned networks using flax.nnx."""

    def __init__(
        self,
        state_encoder: nnx.Module = None,
        goal_encoder: nnx.Module = None,
        concat_encoder: nnx.Module = None,
    ):
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.concat_encoder = concat_encoder

    def __call__(self, observations, goals=None, goal_encoded=False):
        """Returns the representations of observations and goals."""
        reps = []
        
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        
        if goals is not None:
            if goal_encoded:
                # Can't have both goal_encoder and concat_encoder in this case.
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
        
        if len(reps) == 0:
            return observations  # Fallback if no encoders provided
        elif len(reps) == 1:
            return reps[0]
        else:
            return jnp.concatenate(reps, axis=-1)


def create_impala_encoder(rngs: nnx.Rngs, **kwargs):
    """Factory function to create ImpalaEncoder with rngs."""
    return ImpalaEncoder(rngs=rngs, **kwargs)


def create_impala_debug_encoder(rngs: nnx.Rngs):
    """Factory function to create debug IMPALA encoder."""
    return ImpalaEncoder(num_blocks=1, stack_sizes=(4, 4), rngs=rngs)


def create_impala_small_encoder(rngs: nnx.Rngs):
    """Factory function to create small IMPALA encoder."""
    return ImpalaEncoder(num_blocks=1, rngs=rngs)


def create_impala_large_encoder(rngs: nnx.Rngs):
    """Factory function to create large IMPALA encoder."""
    return ImpalaEncoder(
        stack_sizes=(64, 128, 128), 
        mlp_hidden_dims=(1024,),
        rngs=rngs,
    )


# Updated encoder modules dictionary that works with nnx
encoder_modules = {
    'impala': create_impala_encoder,
    'impala_debug': create_impala_debug_encoder,
    'impala_small': create_impala_small_encoder,
    'impala_large': create_impala_large_encoder,
}
