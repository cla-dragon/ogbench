import flax.nnx as nnx
import jax
import jax.numpy as jnp
import distrax

from typing import Any, Optional, Sequence

from impls.utils.encoders_nnx import MLP

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nnx.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class LengthNormalize(nnx.Module):
    """Length normalization layer."""
    
    def __init__(self):
        pass
    
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class Identity(nnx.Module): 
    """Identity layer."""
    
    def __init__(self):
        pass
    
    def __call__(self, x):
        return x


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class GCActor(nnx.Module):
    """Goal-conditioned actor using flax.nnx."""
    
    def __init__(
        self,
        hidden_dims: Sequence[int],
        action_dim: int,
        log_std_min: Optional[float] = -5,
        log_std_max: Optional[float] = 2,
        tanh_squash: bool = False,
        state_dependent_std: bool = False,
        const_std: bool = True,
        final_fc_init_scale: float = 1e-2,
        gc_encoder=None,
        rngs: nnx.Rngs = None,
    ):
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash = tanh_squash
        self.state_dependent_std = state_dependent_std
        self.const_std = const_std
        self.final_fc_init_scale = final_fc_init_scale
        self.gc_encoder = gc_encoder
        self.rngs = rngs
        
        # Create networks with lazy initialization
        self.actor_net = MLP(hidden_dims, activate_final=True, rngs=rngs)
        self.mean_net = None  # Will be initialized on first call
        
        if state_dependent_std:
            self.log_std_net = None  # Will be initialized on first call
        elif not const_std:
            self.log_stds = nnx.Param(jnp.zeros(action_dim))
    
    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution."""
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        
        outputs = self.actor_net(inputs)
        
        # Initialize mean_net on first call
        if self.mean_net is None:
            self.mean_net = nnx.Linear(
                in_features=outputs.shape[-1],
                out_features=self.action_dim,
                kernel_init=default_init(self.final_fc_init_scale),
                rngs=self.rngs,
            )
        
        means = self.mean_net(outputs)
        
        if self.state_dependent_std:
            # Initialize log_std_net on first call
            if self.log_std_net is None:
                self.log_std_net = nnx.Linear(
                    in_features=outputs.shape[-1],
                    out_features=self.action_dim,
                    kernel_init=default_init(self.final_fc_init_scale),
                    rngs=self.rngs,
                )
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds.value
        
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        
        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))
        
        return distribution


class GCDiscreteActor(nnx.Module):
    """Goal-conditioned actor for discrete actions using flax.nnx."""
    
    def __init__(
        self,
        hidden_dims: Sequence[int],
        action_dim: int,
        final_fc_init_scale: float = 1e-2,
        gc_encoder=None,
        rngs: nnx.Rngs = None,
    ):
        self.action_dim = action_dim
        self.final_fc_init_scale = final_fc_init_scale
        self.gc_encoder = gc_encoder
        self.rngs = rngs
        
        self.actor_net = MLP(hidden_dims, activate_final=True, rngs=rngs)
        self.logit_net = None  # Will be initialized on first call
    
    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution."""
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        
        outputs = self.actor_net(inputs)
        
        # Initialize logit_net on first call
        if self.logit_net is None:
            self.logit_net = nnx.Linear(
                in_features=outputs.shape[-1],
                out_features=self.action_dim,
                kernel_init=default_init(self.final_fc_init_scale),
                rngs=self.rngs,
            )
        
        logits = self.logit_net(outputs)
        
        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))
        return distribution


class GCValue(nnx.Module):
    """Goal-conditioned value/critic function using flax.nnx."""
    
    def __init__(
        self,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
        ensemble: bool = True,
        gc_encoder=None,
        rngs: nnx.Rngs = None,
    ):
        self.ensemble = ensemble
        self.gc_encoder = gc_encoder
        
        if ensemble:
            # Create two separate value networks for ensemble
            self.value_net1 = MLP(
                (*hidden_dims, 1), 
                activate_final=False, 
                layer_norm=layer_norm,
                rngs=rngs,
            )
            self.value_net2 = MLP(
                (*hidden_dims, 1), 
                activate_final=False, 
                layer_norm=layer_norm,
                rngs=rngs,
            )
        else:
            self.value_net = MLP(
                (*hidden_dims, 1), 
                activate_final=False, 
                layer_norm=layer_norm,
                rngs=rngs,
            )
    
    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function."""
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        
        if self.ensemble:
            v1 = self.value_net1(inputs).squeeze(-1)
            v2 = self.value_net2(inputs).squeeze(-1)
            return v1, v2
        else:
            v = self.value_net(inputs).squeeze(-1)
            return v