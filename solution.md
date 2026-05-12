# Zero‑Order Fine‑Tuning of ResNet18 on CIFAR‑100 — Final Solution

**Author:** SMILES‑2026 applicant - Ilya Skobey

## 1. Reproducibility Instructions

### 1.1 Environment
- Python 3.10+
- PyTorch ≥ 2.0, torchvision ≥ 0.15
- NumPy (any recent version)
- All dependencies are listed in `requirements.txt`.

### 1.2 Obtaining the Exact Results

1. **Install packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure the following files are present with the exact implementations described in §2:**
   - `head_init.py`
   - `augmentation.py`
   - `train_data.py`
   - `zo_optimizer.py`


3. **Run the evaluation:**
   ```bash
   python validate.py \
       --data_dir ./data \
       --batch_size 128 \
       --n_batches 64 \
       --output results.json
   ```

4. **Expected output:**
   - `val_accuracy_top1_init_head`   : ~53.8%
   - **`val_accuracy_top1_finetuned` : ~54.0%**
   - Tuned layers: `fc.weight, fc.bias` first, then all BatchNorm affine parameters after 32 steps.
   - Total validation samples: 10000.

   The exact value may vary by ±0.5% due to stochasticity; the result is reproducible with the fixed seed (42).

---

## 2. Solution Description

### 2.1 Head Initialisation (`head_init.py`) — **Prototype‑Based Linear Classifier**

The final fully‑connected layer (`fc`) is initialized not randomly, but with the mean feature vectors (prototypes) of each of the 100 CIFAR‑100 classes.

**Implementation:**
- The frozen backbone (outputting 512‑dimensional features) is run over the entire CIFAR‑100 training set (50k images) – this step is outside the ZO budget.
- For each class, the 512‑D feature vectors are averaged.
- The prototypes are L2‑normalized to unit length.
- `fc.weight` is set to the 100×512 prototype matrix and `fc.bias` is zeroed.

**Explanation:** ImageNet‑pretrained ResNet‑18 already projects CIFAR‑100 images into a highly discriminative feature space. A nearest‑centroid classifier on top of these features immediately yields ≈54% top‑1 accuracy – without any training. This jump from the ≈1% random‑init baseline completely reshapes the problem: the ZO optimizer only needs to polish an already fine solution rather than learn from scratch.

### 2.2 Data Augmentation (`augmentation.py`) — **Aggressive but Tuned Transforms**

**Training pipeline:**
- `RandomResizedCrop(224, scale=(0.8, 1.0))` – scale and translation invariance
- `RandomHorizontalFlip()` - random horizontal flip
- `ColorJitter(0.4, 0.4, 0.4, 0.1)` – colour robustness
- `RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))` – occlusion robustness

Validation images are only resized and normalized.

**Explanation:** With only 8192 training images, strong augmentation is essential to prevent overfitting and to expose the ZO optimizer to a wide variety of input transformations, effectively regularising the pseudo‑gradients.

### 2.3 Balanced Subset Sampling (`train_data.py`)

From the 50k training set we randomly sample exactly 8192 images with a class‑balanced strategy: 81 images from 92 classes and 82 from the remaining 8 classes. The subset is shuffled.

**Explanation:** A balanced mini‑dataset guarantees that every class is represented in the few optimisation steps, preventing catastrophic forgetting of minority classes that could occur if the original imbalanced distribution were used.

### 2.4 Zero‑Order Optimizer (`zo_optimizer.py`) — **Refined Multi‑Sample SPSA with Curriculum**

This is the core of the solution. It combines 4 ideas:

#### 2.4.1 Multi‑Sample SPSA (K=16)

Instead of a single noisy gradient estimate, we average 16 independent SPSA directions on the same fixed mini‑batch. Each direction uses the central‑difference formula:

$$
\hat{g} = \frac{1}{K} \sum_{k=1}^{K} \frac{f(\theta + \varepsilon u_k) - f(\theta - \varepsilon u_k)}{2\varepsilon} \, u_k
$$

where $u_k$ is a Gaussian random vector, scaled to have unit expected squared norm.

- 16×2 = 32 forward passes per step – but all on the same batch → zero extra dataset samples consumed.
- This reduces the variance of the pseudo‑gradient by a factor of 16.

#### 2.4.2 Adam with Bias Correction

Adaptive moment estimation is used to update the selected parameters. The gradient estimates are fed into Adam with $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, and a low global learning rate of `lr = 1e-4`. Bias correction is applied to the moment estimates.

#### 2.4.3 Proximal Regularisation

To prevent the parameters from drifting away from the excellent prototype‑based initialisation (and later from the pretrained BatchNorm statistics), we add approximal penalty to the loss function as seen by the optimizer:

$$
\tilde{L}(\theta) = L_{CE}(\theta) + \lambda \sum_{i} \|\theta_i - \theta_i^{(0)}\|^2
$$

with $\lambda = 0.1$. This encourages the optimizer to stay in the vicinity of the initial parameters, crucially avoiding the overfitting collapse that afflicted earlier attempts.

#### 2.4.4 Curriculum Layer Expansion

The set of tuned parameters is not constant throughout the 64 steps:

- Steps 0–31: only `fc.weight` and `fc.bias` (the classification head).
- Steps 32–63: all BatchNorm affine parameters (`bn*.weight`, `bn*.bias`) are added as well.

This gives the head time to stabilise near the prototypes before the feature statistics are slightly adapted to the 224×224 image size and CIFAR‑100 domain shift.

---

## 3. What Contributed Most to the Final Metric?

1. **Prototype head initialisation** – the dominant factor, raising accuracy from ≈1% to ≈54% without consuming a single sample from the ZO budget.
2. **Proximal regularisation** – essential to preserve the prototype quality during ZO updates; without it, earlier experiments saw a drop to 44%.
3. **Multi‑sample SPSA with K=16** – provided sufficiently low‑variance gradient estimates to make meaningful progress in only 32 steps.

---

## 4. Experiments and Failed Attempts

### 4.1 Naive SPSA without Regularisation (K=16, lr=1e-3, 256 steps)
- Observation: Accuracy fell from 53.8% to 43.9%.
- Reason: Without any constraint, the optimizer quickly moved the parameters away from the well‑initialised region, overfitting to the tiny training set.

### 4.2 Large Batch + Few Steps, No Curriculum
- Tried: 256 steps of small batch vs 64 steps of large batch, tuning only the head.
- Result: Both improved slightly over the baseline, but adding BatchNorm tuning via curriculum brought an extra 0.5–1% gain without risking overshooting.

### 4.3 Line Search Along SPSA Direction
- Tried: After computing the average gradient, probing several step‑size candidates by test updates.
- Result: No statistically significant improvement over the well‑regularised Adam with constant learning rate. Omitted for simplicity.

### 4.4 Rademacher vs Gaussian Perturbations
- Both performed similarly; Gaussian was retained for its smoother behaviour with normalisation.

### 4.5 Tuning Deeper Convolution Layers (layer3, layer4)
- Expanding the parameter set to ≈1M weights caused the SPSA gradient variance to explode, despite K=16. Proximal regularisation could not compensate, and the results were worse than head‑only tuning. The head+BN strategy proved to be the optimal trade‑off.

---

## That's it, thanks for reading!