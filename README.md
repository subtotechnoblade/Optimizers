# Optimizers

This repository is a curated implementation of state-of-the-art (SOTA) optimization algorithms and advanced gradient manipulation techniques for deep learning.

While standard optimizers like Adam are powerful, the frontier of deep learning research has produced novel methods that offer faster convergence, better generalization, and more efficient training for large-scale models. This project serves as a clean, modular, and easy-to-use toolkit for researchers and engineers to experiment with these cutting-edge techniques.

The goal is to provide reliable implementations that are a drop in replacement for the default keras optimizer implementation.

##🚀 Project Roadmap
This project is actively developed. Here is the current status of implemented and planned features.


##✅ Implemented Features
Core Optimizers

[x] Adam: The foundational adaptive optimizer that serves as our primary baseline.

[x] Nadam: An integration of Nesterov momentum into the Adam algorithm for often-faster convergence.

[x] Muon Optimizer: A SOTA optimizer for large models. It applies gradient orthogonalization (via Newton-Schulz iteration) specifically to 2D weight matrices, leading to significant efficiency gains and improved stability.


##⏳ On the Workbench (Planned Features)
Gradient Add-ons & Techniques

[ ] GrokFast: A gradient filter designed to accelerate "grokking." It works by amplifying the slow-varying, low-frequency components of the gradients, which are linked to generalization.

[ ] OrthoGrad: A gradient projection technique for continual learning and machine unlearning. It prevents catastrophic forgetting by projecting new gradients into a subspace orthogonal to those of previous tasks.


## 🔧 How to Use
All optimizers in this repo are implemented to be drop in replacement for the default keras optimizers.

###Adam
Adam
