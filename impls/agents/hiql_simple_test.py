"""
简化的 HIQL nnx 训练示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from agents.hiql_nnx import create_hiql_agent, get_config


def test_training():
    """测试训练功能"""
    
    # 配置
    config = get_config()
    config['encoder'] = None  # 不使用视觉编码器
    config['discrete'] = False  # 连续动作
    
    # 创建示例数据
    key = jax.random.PRNGKey(42)
    ex_observations = jax.random.normal(key, (32, 10))
    ex_actions = jax.random.normal(key, (32, 4))
    
    # 创建 agent
    agent = create_hiql_agent(
        seed=42,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )
    
    # 先进行一次前向传播以确保所有层都初始化
    print("初始化所有网络层...")
    
    # 创建测试批次
    batch = {
        'observations': jax.random.normal(key, (64, 10)),
        'next_observations': jax.random.normal(key, (64, 10)),
        'actions': jax.random.normal(key, (64, 4)),
        'rewards': jax.random.normal(key, (64,)),
        'masks': jnp.ones((64,)),
        'value_goals': jax.random.normal(key, (64, 10)),
        'low_actor_goals': jax.random.normal(key, (64, 10)),
        'high_actor_goals': jax.random.normal(key, (64, 10)),
        'high_actor_targets': jax.random.normal(key, (64, 10)),
    }
    
    # 运行一次前向传播以初始化所有层
    loss, info = agent.total_loss(batch)
    print(f"初始损失: {loss:.4f}")
    
    # 现在创建优化器
    print("创建优化器...")
    optimizer = nnx.Optimizer(agent, optax.adam(config['lr']))
    
    # 方法 1: 手动梯度更新
    print("\n方法 1: 手动梯度更新")
    print("-" * 30)
    
    def loss_fn(agent_model):
        return agent_model.total_loss(batch)
    
    # 计算梯度
    (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
    print(f"损失: {loss:.4f}")
    
    # 更新参数
    optimizer.update(grads)
    
    # 更新目标网络
    agent.update_target()
    
    print("训练信息:")
    for key, value in info.items():
        print(f"  {key}: {value:.4f}")
    
    # 方法 2: 使用 agent 的 update 方法
    print("\n方法 2: 使用 agent.update 方法")
    print("-" * 30)
    
    # 重新计算损失验证更新
    loss_after, info_after = agent.total_loss(batch)
    print(f"更新后损失: {loss_after:.4f}")
    
    # 测试动作采样
    print("\n测试动作采样:")
    print("-" * 30)
    obs = jax.random.normal(key, (5, 10))
    goals = jax.random.normal(key, (5, 10))
    actions = agent.sample_actions(obs, goals, seed=key)
    print(f"观测形状: {obs.shape}")
    print(f"目标形状: {goals.shape}")
    print(f"动作形状: {actions.shape}")
    print(f"动作范围: [{actions.min():.3f}, {actions.max():.3f}]")


def test_simple_training_loop():
    """测试简单的训练循环"""
    
    print("\n" + "="*50)
    print("测试训练循环")
    print("="*50)
    
    # 配置
    config = get_config()
    config['encoder'] = None
    config['discrete'] = False
    
    # 创建 agent
    key = jax.random.PRNGKey(42)
    ex_observations = jax.random.normal(key, (32, 10))
    ex_actions = jax.random.normal(key, (32, 4))
    
    agent = create_hiql_agent(
        seed=42,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )
    
    # 初始化
    batch = {
        'observations': jax.random.normal(key, (256, 10)),
        'next_observations': jax.random.normal(key, (256, 10)),
        'actions': jax.random.normal(key, (256, 4)),
        'rewards': jax.random.normal(key, (256,)),
        'masks': jnp.ones((256,)),
        'value_goals': jax.random.normal(key, (256, 10)),
        'low_actor_goals': jax.random.normal(key, (256, 10)),
        'high_actor_goals': jax.random.normal(key, (256, 10)),
        'high_actor_targets': jax.random.normal(key, (256, 10)),
    }
    
    # 前向传播初始化
    _ = agent.total_loss(batch)
    
    # 创建优化器
    optimizer = nnx.Optimizer(agent, optax.adam(config['lr']))
    
    # 训练循环
    num_steps = 3
    for step in range(num_steps):
        
        # 计算梯度和损失
        def loss_fn(agent_model):
            return agent_model.total_loss(batch)
        
        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
        
        # 更新参数
        optimizer.update(grads)
        
        # 更新目标网络
        agent.update_target()
        
        # 打印进度
        print(f"Step {step+1}/{num_steps} | Loss: {loss:.4f} | Value Loss: {info['value/value_loss']:.4f}")
    
    print("训练完成！")


if __name__ == "__main__":
    test_training()
    test_simple_training_loop()
