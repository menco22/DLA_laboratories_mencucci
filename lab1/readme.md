
# Deep Learning Applications — Lab 1  
**Residual Connections and Transfer Learning**

## Abstract
This lab investigates the role of **residual connections** in mitigating vanishing gradients in deep neural networks, both in **MLPs** and **CNNs**, and explores **transfer learning and fine-tuning** for image classification. Starting from a simple MLP on MNIST, we progressively design and analyze convolutional architectures on CIFAR-10, with and without residual blocks. Finally, we transfer a pre-trained residual CNN to CIFAR-100, comparing *linear probing* and different fine-tuning strategies. All experiments are tracked using **Weights & Biases (wandb)** for and monitoring.

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

### 2.5 Improving the chosen model
- I took the 20-DEEP_RESIDUAL_CNN model and tried to improve its learning abilities as much as possible in view of the next exercise.

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
<img width="591" height="489" alt="image" src="https://github.com/user-attachments/assets/02386037-ddd3-4a9c-8379-2f8720d11336" />

As seen before, the deepest the net get, the more the residual connection makes the difference, although in this case we have more of a phenomenon of gradient degradation rather than disappearance.
<img width="2843" height="1493" alt="W B Chart 03_09_2025, 18_56_18" src="https://github.com/user-attachments/assets/69b25d27-08e3-4329-bba9-ad061453251d" />
<img width="2843" height="1493" alt="W B Chart 03_09_2025, 18_56_50" src="https://github.com/user-attachments/assets/df5169a3-ac8c-45df-b4cc-11068ce1904b" />

-**Improved Model** I have achieved good results in improving the performance of the chosen model
<img width="974" height="100" alt="image" src="https://github.com/user-attachments/assets/a1abb275-dbab-44b8-a040-e701b974b14d" />
<img width="3033" height="1593" alt="W B Chart 03_09_2025, 19_07_04" src="https://github.com/user-attachments/assets/ba862ff2-1fe2-4d3b-bca2-43c6284d63b5" />
<img width="3033" height="1593" alt="W B Chart 03_09_2025, 19_07_32" src="https://github.com/user-attachments/assets/f47346c0-ae6d-45a2-915d-6edfe1b49a7f" />

- **Feature Extractor Baseline** I extracted features from the last convolutional layer (before the fully-connected layer) of the pre-trained model on CIFAR-10.
These features were used to train a linear classifier (Linear SVM) on CIFAR-100 to establish a baseline performance. The accuracy obtained with this approach was **41.97%**.

- **Transfer learning** The model performed quite well after pre-training on cifar-10, considering the difficulty of the task on cifar-100.I experimented with different optimizer setups to see which one was best.
Linear Probing:
<img width="1230" height="249" alt="image" src="https://github.com/user-attachments/assets/e0685d18-cf20-448e-88e1-e7c9665530be" />
First Fine Tuning:
<img width="1236" height="258" alt="image" src="https://github.com/user-attachments/assets/3a26c283-e0d2-4583-b48d-81958d249d8c" />
Second Fine Tuning:
<img width="1236" height="251" alt="image" src="https://github.com/user-attachments/assets/59b5fc29-1a59-4917-8794-87cb1effaba8" />
From these results I had intuited that using adam or adamW as the optimizer would be the best strategy.
Third Fine Tuning (only AdamW):
<img width="1223" height="87" alt="image" src="https://github.com/user-attachments/assets/c5e589cb-49f0-4b1c-9205-28fa135fe541" />
We can see clear signs of overfitting, so I tried to improve the model's performance one last time with a more aggressive data augmentation:
<img width="1237" height="89" alt="image" src="https://github.com/user-attachments/assets/84f7a9a2-4128-4084-ba39-db71c05e99cb" />
<img width="3033" height="1593" alt="W B Chart 03_09_2025, 19_19_13" src="https://github.com/user-attachments/assets/f001b102-c47b-4ca5-b9fb-83bbfc230021" />
<img width="3033" height="1593" alt="W B Chart 03_09_2025, 19_19_37" src="https://github.com/user-attachments/assets/9c1b8318-8cf6-43c6-b0bc-df4d8a1ea0b5" />





  
  
