import optax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import ml_collections
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from impls.utils.encoders_nnx import GCEncoder, encoder_modules, MLP

from impls.utils.networks_nnx import LengthNormalize, Identity, GCActor, GCDiscreteActor, GCValue


class HIQLAgent(nnx.Module):
    """Hierarchical implicit Q-learning (HIQL) agent using flax.nnx."""
    
    def __init__(
        self,
        config: ml_collections.ConfigDict,
        ex_observations,
        ex_actions,
        rngs: nnx.Rngs,
    ):
        self.config = config
        
        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]
        
        # Define goal representation network
        goal_rep_layers = []
        if config['encoder'] is not None:
            encoder_factory = encoder_modules[config['encoder']]
            goal_rep_layers.append(encoder_factory(rngs))
        
        goal_rep_layers.extend([
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
                rngs=rngs,
            ),
            LengthNormalize()
        ])
        self.goal_rep = nnx.Sequential(*goal_rep_layers)
        
        # Define encoders
        if config['encoder'] is not None:
            # Pixel-based environments
            encoder_factory = encoder_modules[config['encoder']]
            self.value_encoder = GCEncoder(
                state_encoder=encoder_factory(rngs), 
                concat_encoder=self.goal_rep
            )
            self.target_value_encoder = GCEncoder(
                state_encoder=encoder_factory(rngs), 
                concat_encoder=self.goal_rep
            )
            self.low_actor_encoder = GCEncoder(
                state_encoder=encoder_factory(rngs), 
                concat_encoder=self.goal_rep
            )
            self.high_actor_encoder = GCEncoder(concat_encoder=encoder_factory(rngs))
        else:
            # State-based environments
            self.value_encoder = GCEncoder(
                state_encoder=Identity(), 
                concat_encoder=self.goal_rep
            )
            self.target_value_encoder = GCEncoder(
                state_encoder=Identity(), 
                concat_encoder=self.goal_rep
            )
            self.low_actor_encoder = GCEncoder(
                state_encoder=Identity(), 
                concat_encoder=self.goal_rep
            )
            self.high_actor_encoder = None
        
        # Define networks
        self.value = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=self.value_encoder,
            rngs=rngs,
        )
        
        self.target_value = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=self.target_value_encoder,
            rngs=rngs,
        )
        
        if config['discrete']:
            self.low_actor = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=self.low_actor_encoder,
                rngs=rngs,
            )
        else:
            self.low_actor = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=self.low_actor_encoder,
                rngs=rngs,
            )
        
        self.high_actor = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=self.high_actor_encoder,
            rngs=rngs,
        )
        
        # Copy value parameters to target value
        state_dict = nnx.state(self.value)
        nnx.update(self.target_value, state_dict)
    
    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)
    
    def value_loss(self, batch):
        """Compute the IVL value loss."""
        next_v1_t, next_v2_t = self.target_value(batch['next_observations'], batch['value_goals'])
        next_v1_t = jax.lax.stop_gradient(next_v1_t)
        next_v2_t = jax.lax.stop_gradient(next_v2_t)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t
        
        v1_t, v2_t = self.target_value(batch['observations'], batch['value_goals'])
        v1_t = jax.lax.stop_gradient(v1_t)
        v2_t = jax.lax.stop_gradient(v2_t)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t
        
        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        v1, v2 = self.value(batch['observations'], batch['value_goals'])
        v = (v1 + v2) / 2
        
        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2
        
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }
    
    def low_actor_loss(self, batch):
        """Compute the low-level actor loss."""
        v1, v2 = self.value(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.value(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        v = jax.lax.stop_gradient(v)
        nv = (nv1 + nv2) / 2
        nv = jax.lax.stop_gradient(nv)
        adv = nv - v
        
        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)
        
        # Compute goal representations
        goal_reps = self.goal_rep(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1)
        )
        if not self.config['low_actor_rep_grad']:
            goal_reps = jax.lax.stop_gradient(goal_reps)
        
        dist = self.low_actor(batch['observations'], goal_reps, goal_encoded=True)
        log_prob = dist.log_prob(batch['actions'])
        
        actor_loss = -(exp_a * log_prob).mean()
        
        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update({
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            })
        
        return actor_loss, actor_info
    
    def high_actor_loss(self, batch):
        """Compute the high-level actor loss."""
        v1, v2 = self.value(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.value(batch['high_actor_targets'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        v = jax.lax.stop_gradient(v)
        nv = (nv1 + nv2) / 2
        nv = jax.lax.stop_gradient(nv)
        adv = nv - v
        
        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)
        
        dist = self.high_actor(batch['observations'], batch['high_actor_goals'])
        target = self.goal_rep(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1)
        )
        target = jax.lax.stop_gradient(target)
        log_prob = dist.log_prob(target)
        
        actor_loss = -(exp_a * log_prob).mean()
        
        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }
    
    def total_loss(self, batch):
        """Compute the total loss."""
        info = {}
        
        value_loss, value_info = self.value_loss(batch)
        for k, v in value_info.items():
            info[f'value/{k}'] = v
        
        low_actor_loss, low_actor_info = self.low_actor_loss(batch)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v
        
        high_actor_loss, high_actor_info = self.high_actor_loss(batch)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v
        
        loss = value_loss + low_actor_loss + high_actor_loss
        return loss, info
    
    def update_target(self):
        """Update the target network."""
        # Get current value network parameters (not full state)
        value_graphdef, value_state = nnx.split(self.value)
        target_graphdef, target_state = nnx.split(self.target_value)
        
        # Extract only the parameters for soft update
        value_params = nnx.state(self.value, nnx.Param)
        target_params = nnx.state(self.target_value, nnx.Param)
        
        # Soft update only the parameters
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            value_params,
            target_params
        )
        
        # Update target network parameters
        nnx.update(self.target_value, new_target_params)
    
    def update(self, batch, optimizer):
        """Update the agent and return information dictionary.
        
        Args:
            batch: Training batch data
            optimizer: nnx.Optimizer instance
            
        Returns:
            info: Dictionary containing training information
        """
        
        def loss_fn(agent):
            return agent.total_loss(batch)
        
        # Compute gradients only for trainable parameters
        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self)
        
        # Filter gradients to only include parameters that exist in the optimizer
        # This prevents state mismatch issues
        optimizer_state = nnx.state(optimizer)
        model_params = nnx.state(self, nnx.Param)
        
        # Update parameters using the optimizer
        optimizer.update(grads)
        
        # Update target network
        self.update_target()
        
        # Add the total loss to info
        info['loss'] = loss
        
        return info
    
    @nnx.jit
    def update_jit(self, batch, optimizer):
        """JIT-compiled version of update for better performance."""
        return self.update(batch, optimizer)
    
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        high_seed, low_seed = jax.random.split(seed)
        
        high_dist = self.high_actor(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])
        
        low_dist = self.low_actor(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)
        
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions


def create_hiql_agent(
    seed: int,
    ex_observations,
    ex_actions, 
    config: ml_collections.ConfigDict,
):
    """Create a new HIQL agent using flax.nnx."""
    rngs = nnx.Rngs(seed)
    
    agent = HIQLAgent(
        config=config,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(agent, optax.adamw(learning_rate=config['lr']))
    
    return agent, optimizer


def get_config():
    """Get default configuration for HIQL agent."""
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='hiql_nnx',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=25,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation.
            const_std=True,  # Whether to use constant standard deviation for the actors.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.3,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.7,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
