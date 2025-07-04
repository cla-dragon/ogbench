"""
Training example for HIQL agent using flax.nnx.
This shows how to use the new nnx-based HIQL implementation.
"""
import sys
sys.path.append(".")  # Adjust path to include parent directory
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from hiql_nnx import create_hiql_agent, get_config


class HIQLTrainer:
    """Trainer class for HIQL using flax.nnx."""
    
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        
        # Create optimizer
        self.optimizer = nnx.Optimizer(agent, optax.adam(config['lr']))
    
    def train_step(self, batch):
        """Single training step using agent's update method."""
        return self.agent.update(batch, self.optimizer)
    
    @nnx.jit
    def train_step_jit(self, batch):
        """JIT-compiled training step for better performance."""
        return self.agent.update_jit(batch, self.optimizer)
    
    def train(self, dataset, num_steps=1000):
        """Training loop."""
        for step in range(num_steps):
            batch = dataset.sample(self.config['batch_size'])
            loss, info = self.train_step(batch)
            
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")
                for key, value in info.items():
                    print(f"  {key}: {value:.4f}")


def main():
    """Example usage of the nnx-based HIQL agent."""
    
    # Get configuration
    config = get_config()
    config['encoder'] = None  # No visual encoder for this example
    config['discrete'] = False  # Continuous actions
    
    # Create dummy data
    key = jax.random.PRNGKey(42)
    ex_observations = jax.random.normal(key, (32, 10))  # 32 samples, 10-dim obs
    ex_actions = jax.random.normal(key, (32, 4))  # 4-dim actions
    
    # Create agent
    agent = create_hiql_agent(
        seed=42,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )
    
    # Create trainer
    trainer = HIQLTrainer(agent, config)
    
    # Example batch for training
    batch = {
        'observations': jax.random.normal(key, (1024, 10)),
        'next_observations': jax.random.normal(key, (1024, 10)),
        'actions': jax.random.normal(key, (1024, 4)),
        'rewards': jax.random.normal(key, (1024,)),
        'masks': jnp.ones((1024,)),
        'value_goals': jax.random.normal(key, (1024, 10)),
        'low_actor_goals': jax.random.normal(key, (1024, 10)),
        'high_actor_goals': jax.random.normal(key, (1024, 10)),
        'high_actor_targets': jax.random.normal(key, (1024, 10)),
    }
    
    # Single training step example - Method 1: Direct update
    info = agent.update(batch, trainer.optimizer)
    print(f"Training step completed using agent.update()")
    print("Training info:")
    for key, value in info.items():
        print(f"  {key}: {value:.4f}")
    
    # Single training step example - Method 2: Through trainer
    info2 = trainer.train_step(batch)
    print(f"\nTraining step completed using trainer.train_step()")
    print("Training info:")
    for key, value in info2.items():
        print(f"  {key}: {value:.4f}")
    
    # Sample actions
    obs = jax.random.normal(key, (5, 10))
    goals = jax.random.normal(key, (5, 10))
    actions = agent.sample_actions(obs, goals, seed=key)
    print(f"Sampled actions shape: {actions.shape}")


if __name__ == "__main__":
    main()
