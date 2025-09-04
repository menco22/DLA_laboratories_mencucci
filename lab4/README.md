# Adversarial Robustness and OOD Detection

This project explores adversarial attacks, adversarial training, and out-of-distribution (OOD) detection techniques using CIFAR-10. The exercises are based on implementing Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD, reference), adversarial training, and the ODIN method.

---

## Datasets and Models
 - **Datasets:** CIFAR-10 (in-distribution), fake images or other datasets (out-of-distribution)  
- **Models:**
  - Simple CNN (0.56 test accuracy)
  - Autoencoder (3x32x32 -> 48x4x4 -> 3x32x32)
  - ResNet20 pretrained on CIFAR-10
  - ResNet20 pretrained + adversarially trained with FGSM data augmentation  

---
## Exercise 1 – OOD Detection Pipeline

A **pipeline for out-of-distribution (OOD) detection** was implemented. The goal is to assign a **score to each test sample** reflecting how likely it is to be OOD. The evaluation focuses on both **qualitative and quantitative assessment** of OOD scores to inspect the separation between in-distribution (ID) and out-of-distribution (OOD) data.

### Pipeline Description

1. **Score functions**: Two scoring strategies are used to measure how confident the model is on a given input:  
   - `max_logit`: takes the maximum logit value for each sample.  
   - `max_softmax`: computes the softmax of logits (optionally scaled by temperature `T`) and takes the maximum probability.  

2. **Score computation**:  
   - Each score function is applied to all samples in the data loader using the `compute_scores` function.  
   - This produces a **1D tensor of confidence scores** for each dataset (ID and OOD).  

3. **Visualization**:  
   - **Sorted score plots** show how the confidence values of ID and OOD data are distributed.  
   - **Histograms** provide a density view, allowing visual comparison of the distributions between ID and OOD samples.  

4. **Quantitative evaluation**:  
   - Confidence scores are concatenated for ID and OOD samples to create labels (`1` for ID, `0` for OOD).  
   - **ROC curves** are plotted to assess the model's performance numerically in distinguishing ID from OOD.  

### First Results
 #### Simple CNN
 <div>
  <img src="https://github.com/user-attachments/assets/b7640eae-c95f-4dca-a413-789d304a1251" width="340" style="display:inline-block; margin-right:5px;">
  <img src="https://github.com/user-attachments/assets/3380993b-8b2b-4298-ad1e-bff838ca9f3c" width="340" style="display:inline-block; margin-right:5px;">
  <img src="https://github.com/user-attachments/assets/7e525179-ef03-45ba-b9e8-bcd3092216d1" width="340" style="display:inline-block;">
</div>

---

## Exercise 2.1 – FGSM Attacks

### Implementation
FGSM perturbs inputs along the sign of the gradient of the loss:




- **Untargeted attack:** maximizes loss with respect to true label:

$$
x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x L(\theta, x, y))
$$

- **Targeted attack:** minimizes loss with respect to a chosen target label.

$$
x_{\text{adv}} = x - \epsilon \cdot \text{sign}(\nabla_x L(\theta, x, y_{\text{target}}))
$$

### Results
 ## Targetded 
 Perturbations remain visually imperceptible for small \(\epsilon\), yet predictions change drastically.
  - **Simple CNN** 
  Budget: ε = 13/255
  <img width="416" height="435" alt="download" src="https://github.com/user-attachments/assets/48b9aab1-81bd-4299-9a01-c0b595266e9f" />
  <img width="416" height="435" alt="download" src="https://github.com/user-attachments/assets/f3d79525-369d-4ec2-875a-b1668bc987de" />
  
  Distribution of changes applied to individual pixels:
  
  <img width="560" height="413" alt="download" src="https://github.com/user-attachments/assets/3a930553-577b-4fd2-ad08-e60fa9eb47f7" />

  - **Pre-Trained CNN**
    Budget: ε = 4/255
   <img width="416" height="435" alt="download" src="https://github.com/user-attachments/assets/1089d9ce-3625-4be2-b4c4-540ec049ab8a" />
   <img width="416" height="435" alt="download" src="https://github.com/user-attachments/assets/29fa7311-df74-4ae8-a947-277832738826" />

   Distribution of changes applied to individual pixels:

   <img width="552" height="413" alt="download" src="https://github.com/user-attachments/assets/9f7e8a89-1fd3-4056-80b8-4beb5b3ba10a" />

We can observe that in the case of the pre-trained model, very few pixels underwent significant changes, with perturbations remaining between -0.016 and 0.016 (compared to the previous range of -0.05 and 0.05).Furthermore, the pre-trained model was successfully attacked with a much smaller perturbation—a fact already evident from the budget—compared to the inefficient model trained from scratch. This is likely because the pre-trained model has already learned to recognize the essential features of images, which paradoxically makes it more vulnerable. A more performant model doesn't require large pixel modifications to change its prediction.

 ## Untargeted  
  Pre_Trained model's accuracy drops sharply with increasing ε:

| ϵ (1/255 scale) | Accuracy (%) |
|-----------------|--------------|
| 0               | 83.10        |
| 1               | 55.32        |
| 2               | 35.52        |
| 4               | 20.31        |
| 8               | 14.09        |
| 16              | 11.82        |

<img width="531" height="393" alt="download" src="https://github.com/user-attachments/assets/5bf64d93-cabf-4a06-b88d-47adf43e36de" />

## Exercise 2.2 – Adversarial Training (FGSM)

### Implementation
The model is trained on batches of original and adversarial images (FGSM, ε=1/255) to increase robustness to input perturbations. Improvements in OOD detection can occur as a side effect because the model becomes less overconfident on inputs different from the training data.

### Train Setup
- Perturbation (FGSM): ε = 1/255
- Epochs: 14
- Loss: CrossEntropyLoss
- Optimizer: Adam with learning rate 1e-4

### Evaluation of In-Distribution Robustness
- The model becomes robust to small adversarial perturbations (ε ≤ 2/255)  
- OOD detection improves: the model produces lower confidence on OOD inputs, increasing separability (higher AUROC).
  
<img width="1479" height="490" alt="download" src="https://github.com/user-attachments/assets/0b026219-36a4-4789-8719-6e42d250afa9" />

---

## Exercise 3.1 – ODIN for OOD Detection
Compute confidence scores using ODIN, scaling logits by T and applying a small input perturbation ε to reduce overconfidence on OOD inputs. This improves the separation between ID and OOD data, leading to higher AUROC values, which quantitatively measure the model’s ability to distinguish ID from OOD.

### Results
  #### Simple CNN

  | eps       | T=1       | T=10      | T=100     | T=1000    |
|-----------|-----------|-----------|-----------|-----------|
| 0.000000  | 0.414014  | 0.633222  | 0.662307  | 0.665064  |
| 0.003922  | 0.382092  | 0.618442  | 0.650687  | 0.653753  |
| 0.007843  | 0.354822  | 0.602071  | 0.637345  | 0.640756  |
| 0.015686  | 0.314867  | 0.567371  | 0.607758  | 0.611701  |
| 0.019608  | 0.300875  | 0.549942  | 0.592212  | 0.596382  |
| 0.023529  | 0.289841  | 0.532804  | 0.576594  | 0.580983  |
| 0.039216  | 0.265351  | 0.471823  | 0.517702  | 0.522481  |

<img width="702" height="547" alt="download" src="https://github.com/user-attachments/assets/3dd98ca3-6a2b-42d8-8e30-6232d5849a72" />
  
  #### ResNet20 pretrained

  | eps       | T=1       | T=10      | T=100     | T=1000    |
|-----------|-----------|-----------|-----------|-----------|
| 0.000000  | 0.631797  | 0.621297  | 0.621569  | 0.621587  |
| 0.003922  | 0.883138  | 0.856334  | 0.853637  | 0.853419  |
| 0.007843  | 0.929626  | 0.923397  | 0.921622  | 0.921480  |
| 0.015686  | 0.900511  | 0.920651  | 0.920202  | 0.920185  |
| 0.019608  | 0.860135  | 0.891941  | 0.892418  | 0.892482  |
| 0.023529  | 0.806521  | 0.846623  | 0.847982  | 0.848085  |
| 0.039216  | 0.525699  | 0.568841  | 0.572601  | 0.572835  |

<img width="702" height="547" alt="download" src="https://github.com/user-attachments/assets/31c0e2a1-7be9-4e22-9711-76048f443f8f" />

#### ResNet20 pretrained + FGSM adversarial training

| eps       | T=1       | T=10      | T=100     | T=1000    |
|-----------|-----------|-----------|-----------|-----------|
| 0.000000  | 0.942769  | 0.962699  | 0.962432  | 0.962387  |
| 0.003922  | 0.991680  | 0.990626  | 0.990023  | 0.989959  |
| 0.007843  | 0.997026  | 0.996430  | 0.996163  | 0.996138  |
| 0.015686  | 0.998441  | 0.998561  | 0.998455  | 0.998447  |
| 0.019608  | 0.998430  | 0.998796  | 0.998722  | 0.998713  |
| 0.023529  | 0.998209  | 0.998859  | 0.998808  | 0.998804  |
| 0.039216  | 0.994258  | 0.997834  | 0.997875  | 0.997874  |

<img width="702" height="549" alt="download" src="https://github.com/user-attachments/assets/33b4c803-f805-4901-b098-6681f7040be5" />

---

### Final Comparison

| Model | AUROC baseline (ϵ=0, T=1) | Max AUROC (ODIN) |
|-------|---------------------------|------------------|
| Simple CNN | 0.41 | 0.66 |
| ResNet20 pretrained | 0.63 | ~0.92 |
| ResNet20 + FGSM aug | 0.94 | ~0.999 |

---

