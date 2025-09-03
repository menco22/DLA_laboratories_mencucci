
# Deep Learning Applications — Lab 1  
**Residual Connections and Transfer Learning**

## Abstract
This lab investigates the role of **residual connections** in mitigating vanishing gradients in deep neural networks, both in **MLPs** and **CNNs**, and explores **transfer learning and fine-tuning** for image classification. Starting from a simple MLP on MNIST, we progressively design and analyze convolutional architectures on CIFAR-10, with and without residual blocks. Finally, we transfer a pre-trained residual CNN to CIFAR-100, comparing *linear probing* and different fine-tuning strategies. All experiments are tracked using **Weights & Biases (wandb)** for reproducibility and monitoring.

---

## Objectives
- Analyze gradient propagation in deep **MLPs** with and without residual connections.  
- Design and train **CNNs** (plain vs residual) on CIFAR-10, assessing the effect of skip connections.  
- Apply **transfer learning** from CIFAR-10 to CIFAR-100:
  - Linear probe (frozen backbone).  
  - Fine-tuning strategies (partial vs full).  
- Compare performance across architectures and training strategies.  

---

## Datasets
- **MNIST** (28×28 grayscale, 10 classes) – baseline for MLP.  
- **CIFAR-10** (32×32 RGB, 10 classes) – convolutional architectures.  
- **CIFAR-100** (32×32 RGB, 100 classes) – transfer learning target.  

Data normalization and basic augmentation (random crop, horizontal flip) are applied. Validation splits are carved from the training sets for model selection.

---

## Architectures
### MLP (Baseline on MNIST)
- Class: `oldMLP` — stack of fully-connected layers with ReLU activations.  
- Experiment: depth vs gradient flow, with/without residual shortcut connections.

### CNNs on CIFAR-10
- **Non Residual CNN**: sequential ConvBlocks (Conv3×3 + BN + ReLU).  
- **Residual CNN**: ResidualBlock with identity/projection skip connections (1×1 conv for downsampling).  
- Architecture: Input layer → Residual Blocks (with downsampling at increasing depths) → Global average pooling → Fully connected classifier.

### Transfer Learning Backbone
- Pre-trained deep residual CNN on CIFAR-10.  
- Used as frozen feature extractor (linear probe) or progressively fine-tuned on CIFAR-100.


## Training Pipeline
- **Optimizer**: Adam (default),Adam/AdamW/Sgd/Sgd+momentum.  
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2).  
- **Regularization**: early stopping (patience=5–6).  
- **Logging**: local + wandb (loss, accuracy, gradient flow, checkpoints).  
- **Gradient analysis**: custom functions to plot weight/bias gradient norms across layers, highlighting vanishing/exploding gradients.

---

## Experiments
### 1. Residual Connections in MLP
- Training deep MLPs on MNIST and verification of the effect of the residual connections.

### 2. Residual Connections in CNN
- CIFAR-10 experiments with non-residual vs residual CNNs.   

### 3. Transfer Learning & Fine-tuning on CIFAR-100 (Exercise 2)
- **Feature extractor + Linear SVM**: additional baseline using pre-trained embeddings.  
- **Linear probe (LP)**: freeze backbone, train only final classifier.    
- **Fine-tuning (FT0, FT1, FT2)**: progressively unfreeze layers.    
---

## Results & Discussion
- **Residual connections in MLP** mitigate vanishing gradients:
<img width="942" height="350" alt="Progetto senza titolo" src="https://github.com/user-attachments/assets/d1735013-5a21-4a6b-ad71-84ea177f770e" />
<img width="2844" height="1494" alt="W B Chart 03_09_2025, 18_28_56" src="https://github.com/user-attachments/assets/d775f1c9-e055-4a7d-b691-ac7f0d22d946" />
The deeper the mlp, the more you see the difference.

- **Residual CNNs** clearly outperform plain CNNs in convergence speed and final accuracy:
<img width="666" height="497" alt="image" src="https://github.com/user-attachments/assets/4aff88e6-d1f1-4f7c-887e-4fad4f815ef3" />


    
- **Transfer learning** shows that the model is able to 
