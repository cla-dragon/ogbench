# Encoders 迁移指南: flax.linen → flax.nnx

本文档展示了将 encoders.py 从 `flax.linen` 迁移到 `flax.nnx` 的详细过程。

## 主要变化总览

### 1. ResnetStack 模块

**flax.linen (旧版)**:
```python
class ResnetStack(nn.Module):
    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(features=self.num_features, ...)(x)
        # ... 在 __call__ 中定义所有层
```

**flax.nnx (新版)**:
```python
class ResnetStack(nnx.Module):
    def __init__(self, num_features: int, num_blocks: int, max_pooling: bool = True, rngs: nnx.Rngs = None):
        # 在 __init__ 中预定义所有层
        self.initial_conv = nnx.Conv(...)
        self.blocks = []
        for _ in range(num_blocks):
            block = ResNetBlock(num_features, initializer, rngs=rngs)
            self.blocks.append(block)
    
    def __call__(self, x):
        # 直接使用预定义的层
        conv_out = self.initial_conv(x)
        for block in self.blocks:
            conv_out = block(conv_out)
```

### 2. ImpalaEncoder 模块

**flax.linen (旧版)**:
```python
class ImpalaEncoder(nn.Module):
    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    # ... 其他属性

    def setup(self):
        # 在 setup 中创建子模块
        self.stack_blocks = [ResnetStack(...) for i in range(len(stack_sizes))]
    
    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        # 在这里创建 MLP 等层
        out = MLP(self.mlp_hidden_dims, ...)(out)
```

**flax.nnx (新版)**:
```python
class ImpalaEncoder(nnx.Module):
    def __init__(self, width: int = 1, stack_sizes: tuple = (16, 32, 32), ..., rngs: nnx.Rngs = None):
        # 在 __init__ 中创建所有子模块
        self.stack_blocks = []
        for i in range(len(stack_sizes)):
            stack = ResnetStack(stack_sizes[i] * width, num_blocks, rngs=rngs)
            self.stack_blocks.append(stack)
        
        self.mlp = MLP(mlp_hidden_dims, rngs=rngs)
    
    def __call__(self, x, train=True, cond_var=None):
        # 直接使用预定义的层
        for stack_block in self.stack_blocks:
            conv_out = stack_block(conv_out)
        out = self.mlp(out)
```

### 3. GCEncoder 模块

**flax.linen (旧版)**:
```python
class GCEncoder(nn.Module):
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    concat_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        # 直接使用传入的模块
```

**flax.nnx (新版)**:
```python
class GCEncoder(nnx.Module):
    def __init__(self, state_encoder: nnx.Module = None, goal_encoder: nnx.Module = None, concat_encoder: nnx.Module = None):
        # 存储编码器模块
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.concat_encoder = concat_encoder
    
    def __call__(self, observations, goals=None, goal_encoded=False):
        # 使用存储的编码器
```

### 4. 工厂函数模式

**flax.linen (旧版)**:
```python
encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}

# 使用方式
encoder = encoder_modules['impala']()
```

**flax.nnx (新版)**:
```python
def create_impala_encoder(rngs: nnx.Rngs, **kwargs):
    return ImpalaEncoder(rngs=rngs, **kwargs)

def create_impala_debug_encoder(rngs: nnx.Rngs):
    return ImpalaEncoder(num_blocks=1, stack_sizes=(4, 4), rngs=rngs)

encoder_modules = {
    'impala': create_impala_encoder,
    'impala_debug': create_impala_debug_encoder,
    # ...
}

# 使用方式
encoder_factory = encoder_modules['impala']
encoder = encoder_factory(rngs)
```

## 关键改进

### 1. 更清晰的模块结构
- **nnx**: 所有层在 `__init__` 中定义，结构一目了然
- **linen**: 层在 `@nn.compact` 方法中动态创建，不够直观

### 2. 更好的随机数管理
- **nnx**: 使用 `nnx.Rngs` 统一管理随机数生成器
- **linen**: 依赖于 JAX 的隐式随机数处理

### 3. 简化的子模块创建
- **nnx**: 直接实例化和存储子模块
- **linen**: 需要在 `setup()` 或 `@nn.compact` 中处理

### 4. 更好的工厂模式
- **nnx**: 使用工厂函数，可以传递 `rngs` 参数
- **linen**: 使用 `functools.partial`，但难以处理复杂的初始化需求

## 在 HIQL 中的集成

**更新前的调用**:
```python
from utils.encoders import GCEncoder, encoder_modules

encoder_module = encoder_modules[config['encoder']]
self.value_encoder = GCEncoder(state_encoder=encoder_module(), ...)
```

**更新后的调用**:
```python
from utils.encoders_nnx import GCEncoder, encoder_modules

encoder_factory = encoder_modules[config['encoder']]
self.value_encoder = GCEncoder(state_encoder=encoder_factory(rngs), ...)
```

## 兼容性和优势

1. **完全兼容**: 保持了所有原始功能
2. **更好的类型提示**: nnx 提供了更好的 IDE 支持
3. **更容易调试**: 可以直接检查和修改网络层
4. **更直观的代码**: 更接近传统的深度学习框架风格

## 测试验证

创建了完整的测试套件 `test_encoders_nnx.py` 来验证：
- ✅ ResnetStack 功能正确
- ✅ ImpalaEncoder 各个变体工作正常
- ✅ GCEncoder 正确处理多模态输入
- ✅ 与 HIQL agent 的集成无问题

这个迁移保持了所有原始功能，同时提供了更现代化和易于维护的代码结构。
