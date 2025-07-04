#!/usr/bin/env python3
"""
HIQL nnx Agent Usage Example

This script demonstrates how to use the migrated HIQL agent with flax.nnx
for training and inference.
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np

# Import the nnx version of HIQL
from hiql_nnx import create_hiql_agent, get_config


def create_dummy_batch(batch_size=32, obs_dim=10, action_dim=4, goal_dim=10):
    """Create a dummy batch for testing."""
    return {
        'observations': jnp.ones((batch_size, obs_dim)),
        'next_observations': jnp.ones((batch_size, obs_dim)),
        'actions': jnp.ones((batch_size, action_dim)),
        'rewards': jnp.ones((batch_size,)),
        'masks': jnp.ones((batch_size,)),  # 1 = not done, 0 = done
        'value_goals': jnp.ones((batch_size, goal_dim)),
        'low_actor_goals': jnp.ones((batch_size, goal_dim)),
        'high_actor_goals': jnp.ones((batch_size, goal_dim)),
        'high_actor_targets': jnp.ones((batch_size, goal_dim)),
    }


def main():
    """Main training loop example."""
    
    # Setup
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    # Environment dimensions (example values)
    obs_dim = 10
    action_dim = 4
    goal_dim = 10
    batch_size = 32
    
    # Create example data
    ex_observations = jnp.ones((1, obs_dim))
    ex_actions = jnp.ones((1, action_dim))
    
    # Get configuration
    config = get_config()
    config.encoder = None  # Use state-based (not pixel-based) inputs
    config.discrete = False  # Continuous action space
    
    print("Creating HIQL agent...")
    
    # Create agent
    agent = create_hiql_agent(
        seed=seed,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )
    
    print(f"Agent created successfully!")
    print(f"Agent configuration: {dict(config)}")
    
    # Initialize all layers by running a forward pass
    print("Initializing all network layers...")
    dummy_batch = create_dummy_batch(batch_size, obs_dim, action_dim, goal_dim)
    _ = agent.total_loss(dummy_batch)  # This ensures all lazy layers are initialized
    
    # Create optimizer AFTER initialization
    optimizer = nnx.Optimizer(agent, optax.adam(config.lr))
    
    # Test action sampling
    print("\\nTesting action sampling...")
    test_obs = jnp.ones((5, obs_dim))  # batch of 5 observations
    test_goals = jnp.ones((5, goal_dim))  # batch of 5 goals
    
    actions = agent.sample_actions(
        observations=test_obs,
        goals=test_goals,
        seed=key,
        temperature=1.0
    )
    
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # Training loop example
    print("\\nStarting training example...")
    
    for step in range(5):
        # Create training batch
        batch = create_dummy_batch(batch_size, obs_dim, action_dim, goal_dim)
        
        # Perform training step using manual gradient update
        def loss_fn(agent_model):
            return agent_model.total_loss(batch)
        
        # Compute gradients
        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
        
        # Update parameters
        optimizer.update(grads)
        
        # Update target network
        agent.update_target()
        
        # Add total loss to info
        info['loss'] = loss
        
        print(f"Step {step + 1}:")
        print(f"  Total loss: {info.get('loss', 'N/A'):.4f}")
        print(f"  Value loss: {info.get('value/value_loss', 'N/A'):.4f}")
        print(f"  Low actor loss: {info.get('low_actor/actor_loss', 'N/A'):.4f}")
        print(f"  High actor loss: {info.get('high_actor/actor_loss', 'N/A'):.4f}")
        
        # Test action sampling after training
        if step == 0 or step == 4:  # First and last step
            test_actions = agent.sample_actions(
                observations=test_obs,
                goals=test_goals,
                seed=key,
                temperature=1.0
            )
            print(f"  Sampled actions range: [{test_actions.min():.3f}, {test_actions.max():.3f}]")
    
    print("\\nTraining example completed!")
    
    # Demonstrate JIT compilation
    print("\\nTesting JIT-compiled update...")
    
    # First call will compile
    batch = create_dummy_batch(batch_size, obs_dim, action_dim, goal_dim)
    info_jit = agent.update_jit(batch, optimizer)
    print(f"JIT update - Total loss: {info_jit.get('loss', 'N/A'):.4f}")
    
    # Subsequent calls will be fast
    info_jit = agent.update_jit(batch, optimizer)
    print(f"JIT update (cached) - Total loss: {info_jit.get('loss', 'N/A'):.4f}")
    
    print("\\nAll tests completed successfully!")


def demonstrate_advanced_usage():
    """Demonstrate advanced usage patterns."""
    print("\\n" + "="*50)
    print("ADVANCED USAGE DEMONSTRATION")
    print("="*50)
    
    # Setup
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    obs_dim = 20
    action_dim = 6
    goal_dim = 20
    
    ex_observations = jnp.ones((1, obs_dim))
    ex_actions = jnp.ones((1, action_dim))
    
    # Custom configuration
    config = get_config()
    config.encoder = None
    config.discrete = False
    config.lr = 1e-4  # Lower learning rate
    config.tau = 0.01  # Faster target update
    config.batch_size = 64
    
    # Create agent with custom config
    agent = create_hiql_agent(seed, ex_observations, ex_actions, config)
    
    # Initialize all layers by running a forward pass
    dummy_batch = create_dummy_batch(config.batch_size, obs_dim, action_dim, goal_dim)
    _ = agent.total_loss(dummy_batch)
    
    # Use different optimizer
    optimizer = nnx.Optimizer(agent, optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(config.lr, b1=0.9, b2=0.999)
    ))
    
    print("Custom agent and optimizer created!")
    
    # Batch training with different temperatures
    temperatures = [0.5, 1.0, 2.0]
    
    for temp in temperatures:
        print(f"\\nTesting with temperature: {temp}")
        
        # Sample actions with different temperatures
        test_obs = jnp.ones((3, obs_dim))
        test_goals = jnp.ones((3, goal_dim))
        
        actions = agent.sample_actions(
            observations=test_obs,
            goals=test_goals,
            seed=key,
            temperature=temp
        )
        
        action_std = jnp.std(actions, axis=0).mean()
        print(f"  Action std (temperature {temp}): {action_std:.4f}")
        
        # Training step using manual update
        batch = create_dummy_batch(config.batch_size, obs_dim, action_dim, goal_dim)
        
        def loss_fn(agent_model):
            return agent_model.total_loss(batch)
        
        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
        optimizer.update(grads)
        agent.update_target()
        
        print(f"  Training loss: {loss:.4f}")
    
    print("\\nAdvanced usage demonstration completed!")


if __name__ == "__main__":
    print("HIQL nnx Agent Usage Example")
    print("=" * 40)
    
    # Run basic usage example
    main()
    
    # Run advanced usage example
    demonstrate_advanced_usage()
    
    print("\\n🎉 All examples completed successfully!")
    print("\\nKey features demonstrated:")
    print("- Agent creation with custom configuration")
    print("- Training loop with loss monitoring")
    print("- Action sampling with different temperatures")
    print("- JIT compilation for performance")
    print("- Custom optimizers and advanced configurations")
    print("- Target network updates")
