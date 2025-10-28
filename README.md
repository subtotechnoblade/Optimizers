# Optimizers

This repository is a curated implementation of state-of-the-art (SOTA) optimization algorithms and advanced gradient manipulation techniques for deep learning.

While standard optimizers like Adam are powerful, the frontier of deep learning research has produced novel methods that offer faster convergence, better generalization, and more efficient training for large-scale models. This project serves as a clean, modular, and easy-to-use toolkit for researchers and engineers to experiment with these cutting-edge techniques.

The goal is to provide reliable implementations that are a drop in replacement for the default keras optimizer implementation.

---

## 🚀 Project Roadmap
This project is actively developed. Here is the current status of implemented and planned features.


## ✅ Implemented Features
Core Optimizers

- [x] Adam: The foundational adaptive optimizer that serves as our primary baseline.

- [x] Nadam: An integration of Nesterov momentum into the Adam algorithm for often-faster convergence.

- [x] Muon Optimizer: A SOTA optimizer for large models. It applies gradient orthogonalization (via Newton-Schulz iteration) specifically to 2D weight matrices, leading to significant efficiency gains and improved stability.


## ⏳ On the Workbench (Planned Features)
Gradient Add-ons & Techniques

- [ ] GrokFast: A gradient filter designed to accelerate "grokking". It works by amplifying the slow-varying, low-frequency components of the gradients, which are linked to generalization.

- [ ] OrthoGrad: A gradient projection technique for "grokking". From the Grokking at the Edge of Numerical Stability, orthograd combats the naive loss minimization problem.


# Docs
All optimizers in this repo are implemented to be a drop in replacement for the default keras optimizers.

---

## Muon

---

**Overview:**

The current SOTA optimizer used to train Kimi-K2.
It uses Nesterov momentum along with Newton-Schulz to 
orthogonalize the gradient.

This is used as a hybrid technique. Muon should only be used for hidden weight updates. Embeddings, inputs layers, output heads, bias, gains should
be optimized by standard AdamW.

Currently, this is done by specifying the layers that you want to avoid. Any other excluded parameters (bias, batchnorm, ect.) are
auto-detected and excluded. Note that muon should have a higher learning rate than AdamW, but this is solved in this implementation. 
You are free to tune the adam_lr_ratio, but it most likely won't be necessary. 

**Parameters:**
- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the Muon.
- **`adam_lr_ratio`** *(float, default=1.0)*: The ratio, adam_lr = adam_lr_ratio * muon_lr
- **`weight_decay`** *(float, default=0.004)*: The decoupled weight decay factor.
- **`use_nadam`** *(bool, default=True)*: Uses NadamW instead of AdamW for non-hidden weight updates.
- **`exclude_layers`** *(list[str], default=[])*: A list of layer names to avoid muon updates, uses Adam or Nadam instead.
- **`exclude_embeddings`** *(bool, default=True)*: If True avoided updating embedding layers with Muon, uses Adam or Nadam instead
- **`caution`** *(bool, default=True)*: Applies cautious updates. 
- **`nesterov`** *(bool, default=True)*: Uses Nesterov momentum for Muon
- **`adam_beta_1`** *(float, default=0.90)*: Decay rate for first momentum estimates in Adam or Nadam.
- **`adam_beta_2`** *(float, default=0.995)*: Decay rate for second momentum estimates in Adam or Nadam.
- **`muon_beta`** *(float, default=0.95)*:  Decay rate for muon's momentum estimates.
- **`muon_a`** *(float, default=3.4445)*:  Newton-Schulz a
- **`muon_b`** *(float, default=-4.7750)*:  Newton-Schulz b
- **`muon_c`** *(float, default=2.0315)*:  Newton-Schulz c
- **`ns_steps`** *(int, default=5)*:  Newton-Schulz iterations
- **`epsilon`** *(float, default=1e-8)*:  Epsilon to prevent divide by 0
- **`name`** *(str, default="Muon")*: Optimizer name

**Example Usage:**
```python
import tensorflow as tf
from muon import Muon

optimizer = Muon(
    learning_rate=1e-3,
    adam_lr_ratio=1.0,
    use_nadam=True,
    weight_decay=0.004,
    exclude_layers=["layer_1", "out_layer"],
    exclude_embeddings=True,
    caution=True,
    nesterov=True,
    adam_beta_1=0.9,
    adam_beta_2=0.995,
    muon_beta=0.95)

model = model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", name="layer_1"), # uses Adam/Nadam
    tf.keras.layers.Dense(512, activation="relu"), # hidden layer, uses muon
    tf.keras.layers.Dense(10, name="out_layer") # uses Adam/Nadam
])
model.compile(optimizer=optimizer)
model.fit()
```


## Adam
**Overview:**

---

The classic Adam optimizer which uses the first and second moments of the gradients.
It is enhanced with decoupled weight decay and cautious updates.

**Parameters:**

- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the optimizer.
- **`beta_1`** *(float, default=0.90)*: Decay for the first moment estimates.
- **`beta_2`** *(float, default=0.995)*: Decay for the second moment estimates.
- **`weight_decay`** *(float, default=0.004)*: Decay for the second moment estimates.
- **`cation`** *(bool, default=True)*: Use caution, applies cautious updates for better stability.
- **`epsilon`** *(float, default=1e-8)*: Epsilon to prevent divide by 0.
- **`name`** *(str, default="Adam")*: Optimizer name.

**Example Usage:**
```python
import tensorflow as tf
from adam import Adam

optimizer = Adam(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.995,
    weight_decay=0.004,
    caution=True)

model = create_model()
model.compile(optimizer=optimizer)
model.fit()
```

## Nadam

---

**Overview**

Nadam optimizer is Adam with Nesterov momentum which leads to faster convergence.
It is enhanced by decoupled weight decay and cautious updates.

**Parameters:**
- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the optimizer.
- **`beta_1`** *(float, default=0.90)*: Decay for the first moment estimates.
- **`beta_2`** *(float, default=0.995)*: Decay for the second moment estimates.
- **`weight_decay`** *(float, default=0.004)*: Decay for the second moment estimates.
- **`cation`** *(bool, default=True)*: Use caution, applies cautious updates for better stability.
- **`epsilon`** *(float, default=1e-8)*: Epsilon to prevent divide by 0.
- **`name`** *(str, default="Nadam")*: Optimizer name.
- 
**Example Usage:**
```python
import tensorflow as tf
from nadam import Nadam

optimizer = Nadam(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.995,
    weight_decay=0.004,
    caution=True)

model = create_model()
model.compile(optimizer=optimizer)
model.fit()
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!
If you are interested in implementing one of the "Planned Features,"
please feel free to open an issue to discuss the implementation details or submit a pull request.


