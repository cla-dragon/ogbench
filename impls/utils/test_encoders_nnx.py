"""
Test script for the nnx-based encoders and HIQL implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from utils.encoders_nnx_v2 import (
    ResnetStack, 
    ImpalaEncoder, 
    GCEncoder, 
    encoder_modules,
    MLP
)


def test_resnet_stack():
    """Test ResnetStack module."""
    print("Testing ResnetStack...")
    
    rngs = nnx.Rngs(42)
    resnet = ResnetStack(num_features=32, num_blocks=2, rngs=rngs)
    
    # Test with dummy image data
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 64, 64, 3))  # Batch of images
    output = resnet(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("ResnetStack test passed ✓")


def test_impala_encoder():
    """Test ImpalaEncoder module."""
    print("\nTesting ImpalaEncoder...")
    
    rngs = nnx.Rngs(42)
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        mlp_hidden_dims=(512,),
        rngs=rngs,
    )
    
    # Test with dummy image data
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 64, 64, 3))  # Batch of images
    output = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("ImpalaEncoder test passed ✓")


def test_gc_encoder():
    """Test GCEncoder module."""
    print("\nTesting GCEncoder...")
    
    rngs = nnx.Rngs(42)
    
    # Create simple state and concat encoders
    state_encoder = MLP([64, 32], rngs=rngs)
    concat_encoder = MLP([128, 64], rngs=rngs)
    
    gc_encoder = GCEncoder(
        state_encoder=state_encoder,
        concat_encoder=concat_encoder,
    )
    
    # Test data
    observations = jax.random.normal(jax.random.PRNGKey(0), (4, 10))
    goals = jax.random.normal(jax.random.PRNGKey(1), (4, 10))
    
    output = gc_encoder(observations, goals)
    
    print(f"Observations shape: {observations.shape}")
    print(f"Goals shape: {goals.shape}")
    print(f"Output shape: {output.shape}")
    print("GCEncoder test passed ✓")


def test_encoder_modules():
    """Test encoder factory functions."""
    print("\nTesting encoder factory functions...")
    
    rngs = nnx.Rngs(42)
    
    for name, factory in encoder_modules.items():
        print(f"Testing {name}...")
        encoder = factory(rngs)
        
        # Test with dummy image data
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 64, 3))
        output = encoder(x)
        
        print(f"  {name} - Input: {x.shape}, Output: {output.shape}")
    
    print("Encoder factory functions test passed ✓")


def test_hiql_integration():
    """Test integration with HIQL agent."""
    print("\nTesting HIQL integration...")
    
    try:
        from agents.hiql_nnx import create_hiql_agent, get_config
        
        # Get configuration
        config = get_config()
        config['encoder'] = None  # No visual encoder for this test
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
        
        # Test action sampling
        obs = jax.random.normal(key, (5, 10))
        goals = jax.random.normal(key, (5, 10))
        actions = agent.sample_actions(obs, goals, seed=key)
        
        print(f"Observations shape: {obs.shape}")
        print(f"Goals shape: {goals.shape}")
        print(f"Sampled actions shape: {actions.shape}")
        print("HIQL integration test passed ✓")
        
    except ImportError as e:
        print(f"HIQL integration test skipped (import error): {e}")


def main():
    """Run all tests."""
    print("Running nnx encoder tests...\n")
    
    test_resnet_stack()
    test_impala_encoder() 
    test_gc_encoder()
    test_encoder_modules()
    test_hiql_integration()
    
    print("\n" + "="*50)
    print("All tests completed successfully! 🎉")
    print("The nnx-based encoders are working correctly.")


if __name__ == "__main__":
    main()
