Deep Learning Applications — Lab 1

Residual Connections and Transfer Learning

Abstract

This lab investigates the role of residual connections in mitigating vanishing gradients in deep neural networks, both in MLPs and CNNs, and explores transfer learning and fine-tuning for image classification. Starting from a simple MLP on MNIST, we progressively design and analyze convolutional architectures on CIFAR-10, with and without residual blocks. Finally, we transfer a pre-trained residual CNN to CIFAR-100, comparing linear probing and different fine-tuning strategies. All experiments are tracked using Weights & Biases (wandb) for reproducibility and monitoring.

Objectives

Analyze gradient propagation in deep MLPs with and without residual connections.

Design and train CNNs (plain vs residual) on CIFAR-10, assessing the effect of skip connections.

Apply transfer learning from CIFAR-10 to CIFAR-100:

Linear probe (frozen backbone).

Fine-tuning strategies (partial vs full).

Compare performance across architectures and training strategies.

Datasets

MNIST (28×28 grayscale, 10 classes) – baseline for MLP.

CIFAR-10 (32×32 RGB, 10 classes) – convolutional architectures.

CIFAR-100 (32×32 RGB, 100 classes) – transfer learning target.

Data normalization and basic augmentation (random crop, horizontal flip) are applied. Validation splits are carved from the training sets for model selection.

Architectures
MLP (Baseline on MNIST)

Class: oldMLP — stack of fully-connected layers with ReLU activations.

Experiment: depth vs gradient flow, with/without residual shortcut connections.

CNNs on CIFAR-10

Plain CNN: sequential ConvBlocks (Conv3×3 + BN + ReLU).

Residual CNN: ResidualBlock with identity/projection skip connections (1×1 conv for downsampling).

Architecture: Input layer → Residual Blocks (with downsampling at increasing depths) → Global average pooling → Fully connected classifier.

Transfer Learning Backbone

Pre-trained deep residual CNN on CIFAR-10.

Used as frozen feature extractor (linear probe) or progressively fine-tuned on CIFAR-100.

Training Pipeline

Optimizer: Adam (default), optionally AdamW/SGD.

Scheduler: ReduceLROnPlateau (factor=0.5, patience=2).

Regularization: early stopping (patience=5–6).

Logging: wandb (loss, accuracy, gradient flow, checkpoints).

Gradient analysis: custom functions to plot weight/bias gradient norms across layers, highlighting vanishing/exploding gradients.

Experiments
1. Residual Connections in MLP

Training deep MLPs on MNIST.

Observation: plain MLPs show gradient attenuation; residual connections stabilize training and improve accuracy.

2. Residual Connections in CNN

CIFAR-10 experiments with plain vs residual CNNs.

Gradient norm plots reveal stronger propagation in residual networks.

Residual CNNs achieve higher test accuracy and faster convergence.

3. Transfer Learning & Fine-tuning on CIFAR-100 (Exercise 2)

Linear probe (LP): freeze backbone, train only final classifier.

Expected test accuracy ≈ 35–45%.

Fine-tuning (FT0, FT1, FT2): progressively unfreeze layers.

Full fine-tuning boosts accuracy to ≈ 45–60%, depending on learning rate and augmentation.

Feature extractor + Linear SVM: additional baseline using pre-trained embeddings.

Results & Discussion
