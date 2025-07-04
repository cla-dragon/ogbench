import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
import tensorflow_datasets as tfds
import numpy as np
from typing import Sequence

# ------------------------
# 1. Define the model
# ------------------------
class MLP(nn.Module):
    hidden_sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        x = nn.Dense(10)(x)  # Output layer (10 classes)
        return x

# ------------------------
# 2. Define training state
# ------------------------
class TrainState(train_state.TrainState):
    pass

# ------------------------
# 3. Create data loader
# ------------------------
def get_datasets():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    train_images, train_labels = train_ds['image'], train_ds['label']
    test_images, test_labels = test_ds['image'], test_ds['label']

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    return (train_images, train_labels), (test_images, test_labels)

def get_batch(xs, ys, batch_size, step):
    idx = step * batch_size % xs.shape[0]
    return xs[idx:idx+batch_size], ys[idx:idx+batch_size]

# ------------------------
# 4. Loss and accuracy
# ------------------------
def compute_metrics(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

# ------------------------
# 5. Train and eval step
# ------------------------
@jax.jit
def train_step(state, batch):
    imgs, labels = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, imgs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, labels)
    return state, metrics

@jax.jit
def eval_step(state, batch):
    imgs, labels = batch
    logits = state.apply_fn({'params': state.params}, imgs)
    return compute_metrics(logits, labels)

# ------------------------
# 6. Main training loop
# ------------------------
def main():
    # Data
    (train_images, train_labels), (test_images, test_labels) = get_datasets()

    # Init model and state
    rng = jax.random.PRNGKey(0)
    model = MLP(hidden_sizes=[128, 64])
    params = model.init(rng, jnp.ones([1, 28, 28]))['params']
    tx = optax.adam(1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Training
    batch_size = 128
    num_epochs = 3
    steps_per_epoch = train_images.shape[0] // batch_size

    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            batch = get_batch(train_images, train_labels, batch_size, step)
            state, metrics = train_step(state, batch)
        print(f"Epoch {epoch+1}, Train Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']*100:.2f}%")

    # Evaluation
    test_metrics = eval_step(state, (test_images, test_labels))
    print(f"Test Loss: {test_metrics['loss']:.4f}, Accuracy: {test_metrics['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
