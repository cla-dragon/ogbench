# HIQL Agent: flax.linen vs flax.nnx 迁移指南

本文档展示了将 HIQL agent 从 `flax.linen` 迁移到 `flax.nnx` 的主要变化。

## 主要改进

### 1. 更直观的面向对象编程模型

**flax.linen (旧版)**:
```python
class HIQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)
        
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        
        return self.replace(network=new_network, rng=new_rng), info
```

**flax.nnx (新版)**:
```python
class HIQLAgent(nnx.Module):
    def __init__(self, config, ex_observations, ex_actions, rngs):
        # 直接初始化所有网络组件
        self.value = GCValue(...)
        self.target_value = GCValue(...)
        self.low_actor = GCActor(...)
        self.high_actor = GCActor(...)
    
    def update_target(self):
        # 直接操作网络状态
        value_state = nnx.state(self.value)
        target_state = nnx.state(self.target_value)
        new_target_state = jax.tree_util.tree_map(...)
        nnx.update(self.target_value, new_target_state)
```

### 2. 简化的网络定义

**flax.linen (旧版)**:
```python
class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    
    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
        return x
```

**flax.nnx (新版)**:
```python
class MLP(nnx.Module):
    def __init__(self, hidden_dims, activations=nnx.gelu, activate_final=False, ...):
        self.layers = []
        for i, size in enumerate(hidden_dims):
            layer = nnx.Linear(in_features=None, out_features=size, rngs=rngs)
            self.layers.append(layer)
    
    def __call__(self, x):
        # 直接使用预定义的层
        for i, size in enumerate(self.hidden_dims):
            x = self.layers[layer_idx](x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
        return x
```

### 3. 更简单的训练循环

**flax.linen (旧版)**:
```python
# 复杂的 TrainState 管理
network_def = ModuleDict(networks)
network_tx = optax.adam(learning_rate=config['lr'])
network_params = network_def.init(init_rng, **network_args)['params']
network = TrainState.create(network_def, network_params, tx=network_tx)

# 训练步骤需要手动管理状态
def loss_fn(grad_params):
    return self.total_loss(batch, grad_params, rng=rng)
new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
```

**flax.nnx (新版)**:
```python
# 简单的优化器创建
agent = HIQLAgent(...)
optimizer = nnx.Optimizer(agent, optax.adam(config['lr']))

# 简单的训练步骤
@nnx.jit
def train_step(batch):
    def loss_fn(agent):
        loss, info = agent.total_loss(batch)
        return loss, info
    
    (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
    optimizer.update(grads)
    return loss, info
```

### 4. 参数管理

**flax.linen (旧版)**:
```python
# 复杂的参数复制
params = network.params
params['modules_target_value'] = params['modules_value']

# 目标网络更新
new_target_params = jax.tree_util.tree_map(
    lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
    self.network.params[f'modules_{module_name}'],
    self.network.params[f'modules_target_{module_name}'],
)
network.params[f'modules_target_{module_name}'] = new_target_params
```

**flax.nnx (新版)**:
```python
# 简单的状态复制
state_dict = nnx.state(self.value)
nnx.update(self.target_value, state_dict)

# 简单的目标网络更新
def update_target(self):
    value_state = nnx.state(self.value)
    target_state = nnx.state(self.target_value)
    new_target_state = jax.tree_util.tree_map(
        lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
        value_state, target_state
    )
    nnx.update(self.target_value, new_target_state)
```

## 关键优势

1. **更直观**: nnx 提供了更接近 PyTorch 的编程模型
2. **更简单**: 不需要 `@nn.compact` 装饰器和复杂的状态管理
3. **更灵活**: 可以在运行时动态修改网络结构
4. **更易调试**: 直接访问和修改网络参数
5. **更好的类型提示**: 更好的 IDE 支持和类型检查

## 迁移步骤

1. 将 `flax.linen as nn` 改为 `flax.nnx as nnx`
2. 将 `flax.struct.PyTreeNode` 改为 `nnx.Module`
3. 移除 `@nn.compact` 装饰器，在 `__init__` 中定义所有层
4. 使用 `nnx.Optimizer` 替代手动的梯度更新
5. 使用 `nnx.state()` 和 `nnx.update()` 进行状态管理
6. 简化训练循环和参数管理

这个新的实现保持了所有原始功能，但提供了更现代化和易于使用的 API。
