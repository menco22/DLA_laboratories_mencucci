# Adversarial Robustness and OOD Detection

This project explores adversarial attacks, adversarial training, and out-of-distribution (OOD) detection techniques using CIFAR-10. The exercises are based on implementing Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD, reference), adversarial training, and the ODIN method.

---

## Datasets and Models
 - **Datasets:** CIFAR-10 (in-distribution), fake images or other datasets (out-of-distribution)  
- **Models:**
  - Simple CNN (baseline)
  - Autoencoder
  - ResNet20 pretrained on CIFAR-10
  - ResNet20 pretrained + adversarially trained with FGSM data augmentation  

---

## Exercise 1 – OOD Detection Pipeline (Baseline)

### Implementation
The goal is to separate in-distribution (ID) and out-of-distribution (OOD) samples using **softmax confidence scores**.  
- For each input, compute the maximum softmax probability.  
- Use this value as the OOD score: higher = more likely ID, lower = more likely OOD.  
- Evaluate detection with ROC curve and AUROC metric.

### Results
- **CDF and histogram plots** show clear separation between ID and OOD scores for strong models.  
- **AUROC scores:**
  - Simple CNN: ~0.55 (slightly better than random)  
  - ResNet20 pretrained: ~0.80  
  - ResNet20 pretrained + FGSM aug: ~0.94  

### Key Takeaways
- A naïve OOD detector based on confidence already works for sufficiently strong models.  
- Weak models tend to be overconfident on OOD data, leading to poor separation.  
- This exercise provides the baseline for evaluating improvements with ODIN later.

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
- **Qualitative:**  
  Perturbations remain visually imperceptible for small \(\epsilon\), yet predictions change drastically.  
- **Quantitative:**  
  Model accuracy drops sharply with increasing \(\epsilon\):

| ϵ (1/255 scale) | Accuracy (%) |
|-----------------|--------------|
| 0               | 83.10        |
| 1               | 55.32        |
| 2               | 35.52        |
| 4               | 20.31        |
| 8               | 14.09        |
| 16              | 11.82        |

### Key Takeaways
- Pretrained models are more vulnerable: they can be fooled with very small perturbations.  
- Robustness and accuracy are in tension: higher accuracy often leads to lower robustness.  

---

## Exercise 2.2 – Adversarial Training (FGSM)

### Implementation
During training, adversarial examples are generated on-the-fly using FGSM with \(\epsilon=1/255\). Both clean and adversarial images are used in each batch.  

### Results
- The model becomes robust to small adversarial perturbations (ϵ ≤ 2/255).  
- Slight trade-off: clean accuracy may decrease compared to standard training.  
- OOD detection improves: the model produces lower confidence on OOD inputs, increasing separability (higher AUROC).  

### Key Takeaways
- Adversarial training is effective at the perturbation level it is trained on.  
- Generalization to stronger attacks (e.g., PGD) is limited if only FGSM is used.  

---

## Exercise 3.1 – ODIN for OOD Detection

ODIN improves OOD detection using:  
1. **Temperature scaling (T):** sharpens or smooths softmax probabilities.  
2. **Small input perturbation (ϵ):** moves ID samples toward higher confidence predictions.  

### Results

#### ResNet20 pretrained + FGSM adversarial training
- AUROC baseline: 0.94  
- Peak AUROC: ~0.999 for ϵ ≈ 0.015–0.02  
- Very stable across T = 1–1000  

#### Simple CNN
- AUROC baseline: 0.41 (worse than random)  
- Max AUROC: ~0.66 with ODIN  
- ODIN ineffective for weak models  

#### ResNet20 pretrained (no adversarial augmentation)
- AUROC baseline: 0.63  
- Peak AUROC: ~0.92 for ϵ ≈ 0.007–0.015  
- Declines for ϵ > 0.02  

### Key Takeaways
- ODIN significantly improves OOD detection for strong models but not for weak ones.  
- ϵ is the most important parameter: small values (~0.007–0.02) yield the best results.  
- Temperature T has secondary impact compared to ϵ.  
- Adversarial training enhances both robustness and OOD detection.  

---

## Final Comparison

| Model | AUROC baseline (ϵ=0, T=1) | Max AUROC (ODIN) |
|-------|---------------------------|------------------|
| Simple CNN | 0.41 | 0.66 |
| ResNet20 pretrained | 0.63 | ~0.92 |
| ResNet20 + FGSM aug | 0.94 | ~0.999 |

---

## Conclusions
1. Exercise 1 established a baseline OOD detector based on softmax confidence.  
2. Exercise 2.1 showed that small, imperceptible perturbations can drastically reduce accuracy.  
3. Exercise 2.2 demonstrated that adversarial training improves robustness and OOD separation.  
4. Exercise 3.1 confirmed that ODIN is highly effective for strong models, especially when combined with adversarial training.  
5. Model capacity and training strategy are crucial: weak models remain ineffective regardless of ODIN.  
