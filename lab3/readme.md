# Sentiment Analysis with DistilBERT on Rotten Tomatoes  

## Theoretical Background  

### Transformers  
Transformers are the current state-of-the-art in deep learning for NLP and vision tasks. Their key innovation is the **self-attention mechanism**, which enables the model to weigh relationships between tokens dynamically, regardless of their position in the sequence.

Key components:  
- **Input/Output embeddings**: map tokens to continuous vector representations.  
- **Positional encoding**: injects information about token order.  
- **Multi-head self-attention**: captures dependencies across different representational subspaces.  
- **Feed-forward layers and residual connections**: increase representational capacity and stability.  

### LoRA (Low-Rank Adaptation)  
LoRA is an efficient fine-tuning method for large models. Instead of updating all parameters, it introduces low-rank matrices \(A\) and \(B\) into selected layers. The updated weight becomes:  

\[
W' = W + BA
\]  

Where \(W\) is the frozen weight matrix and \(BA\) is the low-rank update learned during fine-tuning.  

Benefits:  
- **Drastic reduction of trainable parameters**.  
- **Lower memory and computational cost**.  
- **Reusable adapters**: the same base model can support different tasks with different LoRA adapters.  

### Mixed Precision Training  
Mixed precision combines **FP16 (half precision)** and **FP32 (single precision)** during training.  
- Most matrix multiplications are executed in FP16.  
- Critical parameters and gradients are kept in FP32 for stability.  

Benefits:  
- **Up to 50% less GPU memory usage**.  
- **Faster training** due to specialized hardware (Tensor Cores).  
- **Larger batch sizes** possible under the same memory limits.  

---

## Dataset  

The dataset used is available on HuggingFace:  
`cornell-movie-review-data/rotten_tomatoes`.  

Available splits:  
- **Train**: 8,530 samples  
- **Validation**: 1,066 samples  
- **Test**: 1,066 samples  

Each sample contains:  
- **text**: the movie review  
- **label**: 0 (negative), 1 (positive)  

---

## Exercise 1: Stable Baseline  

### 1.1 - Dataset Splits  
Exploration of splits of the dataset  

### 1.2 - Pre-trained DistilBERT  
Loading the model and tokenizer, verifying embeddings and the last_hidden_state.  

### 1.3 - Feature Extraction + SVM  
- Extract embeddings from the [CLS] token using the `feature-extraction` pipeline.  
- Train a Linear SVM on the training set.  

**Baseline Results (SVM):**  
- Validation Accuracy: 0.82  
- Test Accuracy: 0.80  

---

## Exercise 2: Fine-tuning DistilBERT  

### 2.1 - Token Preprocessing  
Tokenization of the dataset using `Dataset.map`.  

### 2.2 - Model Preparation  
Use of `AutoModelForSequenceClassification` with `num_labels=2`.  

### 2.3 - Fine-tuning with HuggingFace Trainer  
- **Head-only fine-tuning** (frozen backbone).  
- **Full fine-tuning** (all parameters trainable).  

**Results:**  

- **Head-only Fine-tuning**  
  - Validation Accuracy: ~0.78  
  - Validation F1: ~0.77  

- **Full Fine-tuning**  
  - Validation Accuracy: ~0.80  
  - Validation F1: ~0.80  

---

## Exercise 3: Efficient Fine-tuning with LoRA  

Full fine-tuning of DistilBERT is computationally expensive. In this exercise, efficient fine-tuning techniques are explored with **LoRA (Low-Rank Adaptation)** and **mixed precision (fp16)** using the PEFT library.  

### 3.1 - LoRA Setup  
- Target modules: `q_lin` and `v_lin` (DistilBERT attention).  
- LoRA parameters: `r=8, alpha=16, dropout=0.1`.  
- Mixed precision enabled (`fp16=True`).  

**Results:**  

- **LoRA (lr=2e-5)**  
  - Validation Accuracy: ~0.817  
  - Validation F1: ~0.812  

- **LoRA (lr=2e-4)**  
  - Validation Accuracy: ~0.841  
  - Validation F1: ~0.840  

---

## Conclusions  

- The baseline SVM with DistilBERT embeddings already provides strong results (~82% validation accuracy).  
- Fine-tuning DistilBERT slightly improves performance (~80% validation accuracy).  
- LoRA with mixed precision achieves the best results (~84% validation accuracy, F1 ~0.84) with far fewer trainable parameters (739k vs >66M).  
- LoRA proves to be an effective and efficient fine-tuning method, achieving competitive or superior results at a fraction of the cost.  

