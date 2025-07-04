"""
完整的训练示例：展示 HIQL nnx 版本的不同使用方式
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from agents.hiql_nnx import create_hiql_agent, get_config


def create_dummy_batch(batch_size=1024, obs_dim=10, action_dim=4):
    """创建虚拟训练数据"""
    key = jax.random.PRNGKey(42)
    
    batch = {
        'observations': jax.random.normal(key, (batch_size, obs_dim)),
        'next_observations': jax.random.normal(key, (batch_size, obs_dim)),
        'actions': jax.random.normal(key, (batch_size, action_dim)),
        'rewards': jax.random.normal(key, (batch_size,)),
        'masks': jnp.ones((batch_size,)),
        'value_goals': jax.random.normal(key, (batch_size, obs_dim)),
        'low_actor_goals': jax.random.normal(key, (batch_size, obs_dim)),
        'high_actor_goals': jax.random.normal(key, (batch_size, obs_dim)),
        'high_actor_targets': jax.random.normal(key, (batch_size, obs_dim)),
    }
    return batch


def demo_training_methods():
    """演示不同的训练方法"""
    
    # 配置和创建 agent
    config = get_config()
    config['encoder'] = None
    config['discrete'] = False
    
    key = jax.random.PRNGKey(42)
    ex_observations = jax.random.normal(key, (32, 10))
    ex_actions = jax.random.normal(key, (32, 4))
    
    agent = create_hiql_agent(
        seed=42,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )
    
    batch = create_dummy_batch()
    
    print("=== HIQL nnx 训练方法演示 ===\n")
    
    # 方法 1: 直接使用 agent.update()
    print("方法 1: 直接使用 agent.update()")
    print("-" * 40)
    
    optimizer = nnx.Optimizer(agent, optax.adam(config['lr']))
    info = agent.update(batch, optimizer)
    
    print("训练信息:")
    for key, value in info.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # 方法 2: 使用 HIQLTrainer 类
    print("方法 2: 使用 HIQLTrainer 类")
    print("-" * 40)
    
    from agents.hiql_nnx_example import HIQLTrainer
    
    # 重新创建 agent 用于演示
    agent2 = create_hiql_agent(
        seed=43,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )
    
    trainer = HIQLTrainer(agent2, config)
    info2 = trainer.train_step(batch)
    
    print("训练信息:")
    for key, value in info2.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # 方法 3: 手动梯度计算（最底层方式）
    print("方法 3: 手动梯度计算")
    print("-" * 40)
    
    # 重新创建 agent 用于演示
    agent3 = create_hiql_agent(
        seed=44,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )
    
    optimizer3 = nnx.Optimizer(agent3, optax.adam(config['lr']))
    
    def loss_fn(agent):
        return agent.total_loss(batch)
    
    # 计算梯度
    (loss, info3), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent3)
    
    # 更新参数
    optimizer3.update(grads)
    
    # 更新目标网络
    agent3.update_target()
    
    print(f"总损失: {loss:.4f}")
    print("训练信息:")
    for key, value in info3.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # 方法 4: JIT 编译版本（推荐用于生产环境）
    print("方法 4: JIT 编译版本（性能最佳）")
    print("-" * 40)
    
    # 重新创建 agent 用于演示
    agent4 = create_hiql_agent(
        seed=45,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )
    
    optimizer4 = nnx.Optimizer(agent4, optax.adam(config['lr']))
    
    # 第一次调用会进行编译，后续会很快
    print("首次调用 (包含编译时间)...")
    info4 = agent4.update_jit(batch, optimizer4)
    
    print("训练信息:")
    for key, value in info4.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n再次调用 (已编译，速度很快)...")
    info5 = agent4.update_jit(batch, optimizer4)
    
    print("训练信息:")
    for key, value in info5.items():
        print(f"  {key}: {value:.4f}")


def demo_training_loop():
    """演示完整的训练循环"""
    
    print("\n" + "="*60)
    print("演示训练循环")
    print("="*60 + "\n")
    
    # 创建 agent
    config = get_config()
    config['encoder'] = None
    config['discrete'] = False
    
    key = jax.random.PRNGKey(42)
    ex_observations = jax.random.normal(key, (32, 10))
    ex_actions = jax.random.normal(key, (32, 4))
    
    agent = create_hiql_agent(
        seed=42,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )
    
    optimizer = nnx.Optimizer(agent, optax.adam(config['lr']))
    
    # 训练循环
    num_steps = 5
    for step in range(num_steps):
        batch = create_dummy_batch()
        info = agent.update(batch, optimizer)
        
        if step % 1 == 0:  # 每步都打印
            print(f"Step {step:3d} | ", end="")
            print(" | ".join([f"{k}: {v:.4f}" for k, v in list(info.items())[:3]]))
    
    print(f"\n训练完成！共 {num_steps} 步")
    
    # 测试动作采样
    obs = jax.random.normal(key, (5, 10))
    goals = jax.random.normal(key, (5, 10))
    actions = agent.sample_actions(obs, goals, seed=key)
    print(f"动作采样测试: {actions.shape}")


if __name__ == "__main__":
    demo_training_methods()
    demo_training_loop()
