## Abstract
Questo laboratorio esplora la costruzione, l’addestramento e l’analisi di **reti neurali profonde** su dataset di visione artificiale. Il notebook guida dalla **baseline MLP su MNIST** alla **CNN (con e senza connessioni residue) su CIFAR-10**, fino al **transfer learning su CIFAR-100** tramite *linear probing* e *fine-tuning*. Sono inclusi strumenti per **ispezionare il flusso del gradiente**, **monitorare gli esperimenti con Weights & Biases (wandb)** e **riutilizzare le feature** per classificatori lineari (SVM).

---

## Obiettivi didattici
- Costruire una baseline **MLP** e comprenderne limiti su dati d’immagine.
- Progettare una **CNN moderna** con *ConvBlock* e **ResidualBlock** (downsampling con `stride=2` e proiezione 1×1).
- Analizzare l’**attenuazione del gradiente** con/senza skip-connection (*gradient flow*).
- Implementare una routine di **training** con *early stopping*, **scheduler ReduceLROnPlateau**, logging su **wandb**.
- Eseguire **transfer learning**: *linear probe* e **fine-tuning** progressivo su **CIFAR-100** partendo da una CNN addestrata su **CIFAR-10**.
- (Opzionale) Impostare **knowledge distillation** (teacher→student) e **explainability** (Grad-CAM/saliency).

---

## Dataset
- **MNIST** (28×28, 1 canale, 10 classi) — usato per l’introduzione/MLP.
- **CIFAR-10** (32×32, RGB, 10 classi) — usato per CNN e analisi del gradiente.  
  Normalizzazione: `mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2470, 0.2435, 0.2616)`.
- **CIFAR-100** (32×32, RGB, 100 classi) — usato per transfer learning.  
  Normalizzazione: `mean=(0.5071, 0.4867, 0.4408)`, `std=(0.2675, 0.2565, 0.2761)`.  
  Data augmentation (train): *RandomCrop*, *RandomHorizontalFlip* (più eventuali jitter), split di validazione da train (5.000 campioni).

Batch size tipico: **64**. Dataloader per train/val/test con `num_workers=4`.

---

## Architetture implementate

### Baseline MLP (MNIST)
- Classe: `oldMLP(layer_sizes)` → catena di layer **Linear** con ReLU (costruita in modo funzionale).  
- Configurazione usata nel notebook: `[784] + [16]*2 + [10]` (width=16, depth=2).

### CNN non residua (CIFAR-10)
- **ConvBlock**: due conv 3×3 + BatchNorm + ReLU, senza skip-connection.
- Struttura: `input_layer` → sequenza di `ConvBlock` → `AdaptiveAvgPool2d(1)` → `fc`.

### CNN residua profonda
- **ResidualBlock**: due conv 3×3 + BN; **skip** identità o proiezione 1×1 se cambia risoluzione/canali (`stride=2`).  
- Classe principale (es.): `DeepResidualCNN(input_channels, num_classes, depth, base_channels)` con:
  - `input_layer`: conv 3×3, BN, ReLU.
  - `blocks`: pila di `ResidualBlock` con posizionamento di downsampling automatizzato.
  - `global_pool`: `AdaptiveAvgPool2d((1,1))`.
  - `fc`: classificatore lineare finale.
- Iperparametri tipici dal notebook: `depth=20`, `base_channels=32`.

---

## Routine di training e monitoraggio
- Funzione generica: `train_wandb(model, criterion, optimizer, scheduler, num_epochs, device, trainloader, valloader, early_stopping, patience, delta, use_wandb=True)`
  - Log di **loss/accuracy** per batch ed epoch, tempo per batch, best model tracking.
  - **Early stopping** con `patience` e `delta`.
  - Scheduler: **ReduceLROnPlateau** (mode=`min`, `factor=0.5`, `patience=2`, `min_lr=1e-6`).
- Ottimizzatori testati: **Adam** (default), alternative commentate (**SGD** con momentum, **AdamW**).  
- Integrazione **wandb**: `setup_wandb(...)`, `save_model_to_wandb(...)` (esporta `*.pth` come *artifact*).

> **Nota sicurezza**: nel notebook compare una chiave `wandb.login(key=...)`. **Non** committare chiavi API; usare `wandb login` da CLI o variabili d’ambiente.

---

## Esperimenti (come presentati nel notebook)

### Esercizio 1.0–1.1 — Setup e baseline MLP su MNIST
- Addestramento MLP compatto per verificare pipeline, *dataloading* e logging.  
  Iperparametri tipici: `epochs=20`, `lr=1e-4` (senza early stopping).

### Esercizio 1.2 — Residual connections e *gradient flow*
- Implementazione dei residui (versione MLP/CNN) e **analisi delle norme del gradiente**.  
- Funzioni: `gradient_magnitudes_plot(...)`, `gradient_magnitudes_plot_cnn(...)` per visualizzare l’attenuazione/propagazione.

### Esercizio 1.3 — CNN su CIFAR-10
- Addestramento CNN (non-residua e residua) su CIFAR-10.  
- Obiettivo: stabilire una baseline più forte del MLP e confrontare gradienti.

### Esercizio 2.1 — *Fine-tune* su CIFAR-100
**Fase 0: Linear Probe (LP)**  
- Congela il backbone (`requires_grad=False` eccetto `fc`); addestra **solo** il classificatore finale.  
- Esecuzione salvata come: `LP_20_deep_residual_cnn.pth` (artifacts wandb).

**Fase 1–2: Fine-tuning progressivo (FT0/FT1/FT2)**  
- Scongela progressivamente i blocchi (strategie FT0, FT1, FT2 nel notebook).  
- Iperparametri tipici: `epochs=20`, `lr∈{1e-4, 1e-2}`, **ReduceLROnPlateau**, *early stopping* (`patience∈{5,6}`).  
- Esecuzioni salvate come: `FT{0,1,2}_20_deep_residual_cnn.pth`.
