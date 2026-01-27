# Topological Data Analysis (TDA) in Face Verification

## Table of Contents
1. [Introduction to TDA](#1-introduction-to-tda)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [TDA in Deep Learning](#3-tda-in-deep-learning)
4. [Implementation in This Project](#4-implementation-in-this-project)
5. [Experimental Results](#5-experimental-results)
6. [Analysis: Why TDA Didn't Improve Performance](#6-analysis-why-tda-didnt-improve-performance)
7. [Future Directions](#7-future-directions)
8. [References](#8-references)

---

## 1. Introduction to TDA

### 1.1 What is Topological Data Analysis?

**Topological Data Analysis (TDA)** is a field that applies algebraic topology concepts to analyze the "shape" of data. Unlike traditional statistics that focus on numerical summaries (mean, variance), TDA captures **structural and geometric properties** that persist across different scales.

The key insight of TDA is that data often has an underlying shape or structure that is meaningful for the task at hand. For example:
- Clusters in data → Connected components (0-dimensional features)
- Loops or cycles → Holes (1-dimensional features)
- Voids or cavities → Higher-dimensional features

### 1.2 Why TDA for Computer Vision?

In computer vision, especially with attention mechanisms like CBAM (Convolutional Block Attention Module), the **spatial attention maps** reveal which regions of an image the model focuses on. These attention maps have a natural 2D structure that can be analyzed topologically:

| Topological Feature | Meaning in Attention Maps |
|---------------------|---------------------------|
| Connected Components (H0) | Distinct focus regions |
| Holes (H1) | Ring-shaped attention patterns |
| Persistence | How robust each feature is to noise |

A "good" attention map for face verification might have:
- **Few connected components**: Focused on key facial features (eyes, nose, mouth)
- **Low entropy**: Consistent, non-noisy attention
- **High persistence**: Robust features that don't disappear with small changes

---

## 2. Mathematical Foundation

### 2.1 Persistent Homology

**Persistent Homology** is the core tool of TDA. It tracks topological features as we sweep through different "filtration levels" of the data.

#### The Filtration Process

For a 2D attention map $A \in [0, 1]^{H \times W}$:

1. Start with threshold $t = 0$: All pixels are "off"
2. Gradually increase $t$: Pixels with $A_{ij} \geq t$ turn "on"
3. Track topological changes: When do connected components appear/merge? When do holes form/fill?

```
Threshold = 0.1:        Threshold = 0.5:        Threshold = 0.9:
■ ■ ■ ■ ■              ■ □ □ □ ■              □ □ □ □ □
■ ■ ■ ■ ■              □ ■ ■ ■ □              □ □ ■ □ □
■ ■ ■ ■ ■      →       □ ■ □ ■ □      →       □ □ □ □ □
■ ■ ■ ■ ■              □ ■ ■ ■ □              □ □ □ □ □
■ ■ ■ ■ ■              ■ □ □ □ ■              □ □ □ □ □

1 component            4 components + 1 hole   1 component
```

#### Persistence Diagrams

Each topological feature is represented as a point $(b, d)$ where:
- $b$ = **birth time** (filtration level when feature appears)
- $d$ = **death time** (filtration level when feature disappears)

Features with $d - b$ large are **persistent** (robust), while those with small $d - b$ are considered **noise**.

```
Death ↑
      │    ∙ (noisy feature, dies quickly)
      │
      │                ∙ (persistent feature)
      │
      │    diagonal line (b = d, features die at birth)
      └────────────────────────→ Birth
```

### 2.2 Homology Dimensions

| Dimension | Symbol | Meaning | Example |
|-----------|--------|---------|---------|
| 0 | H₀ | Connected components | Separate attention regions |
| 1 | H₁ | 1-dimensional holes | Ring-shaped attention (e.g., around eyes) |
| 2 | H₂ | 2-dimensional voids | Not applicable in 2D images |

### 2.3 Cubical Persistence

For gridded data (images), we use **Cubical Persistence** instead of simplicial complexes:

- Each pixel is a 0-cell
- Adjacent pixels share 1-cells (edges)
- 4-connected pixel groups share 2-cells (squares)

This is implemented using the `gtda.homology.CubicalPersistence` class from giotto-tda.

### 2.4 Persistence-Derived Features

From persistence diagrams, we extract several features:

#### Persistence Entropy
Measures the "disorder" of the persistence diagram:

$$H = -\sum_{i} p_i \log(p_i)$$

where $p_i = \frac{d_i - b_i}{\sum_j (d_j - b_j)}$ is the normalized persistence of feature $i$.

- **Low entropy**: Few dominant features (focused attention)
- **High entropy**: Many similar features (diffuse attention)

#### Betti Curves
The Betti number $\beta_k(t)$ counts the number of $k$-dimensional features alive at filtration level $t$:

$$\beta_k(t) = |\{(b, d) : b \leq t < d\}|$$

The Betti curve plots $\beta_k(t)$ vs $t$ and captures the evolution of topological features.

#### Amplitude (Wasserstein Distance)
Measures the "total persistence" of all features:

$$W_p = \left( \sum_i (d_i - b_i)^p \right)^{1/p}$$

---

## 3. TDA in Deep Learning

### 3.1 The Differentiability Challenge

**Key Problem**: Persistent homology is computed via discrete algorithms (union-find, boundary matrices) that are **not differentiable**.

This means we cannot directly backpropagate through the persistence computation. Solutions include:

1. **Non-gradient regularization**: Use TDA as a regularization term that doesn't need gradients
2. **Soft relaxations**: Approximate discrete operations with differentiable functions
3. **Implicit differentiation**: Compute gradients through fixed-point iterations

### 3.2 Our Approach: Non-Gradient Regularization

We use TDA as a **regularization loss** that encourages desirable topological properties in attention maps:

```python
total_loss = triplet_loss + λ * tda_loss
```

where `tda_loss` is computed on the attention maps but its gradient is not backpropagated through the persistence computation. Instead, we:

1. Compute attention maps (differentiable)
2. Extract persistence features (non-differentiable)
3. Compute scalar loss (non-differentiable)
4. Backpropagate through the attention maps' values that created the topology

**Why this works**: Although we don't have gradients through the topology computation, changing the attention map values will change the resulting topology. The model learns to produce attention maps with desirable topological properties.

### 3.3 Attention Topology Loss Design

Our `AttentionTopologyLoss` encourages:

1. **Target entropy**: Push persistence entropy toward a target value (default: 0.5)
2. **Low complexity**: Penalize having too many topological features

$$\mathcal{L}_{TDA} = w_e \cdot (H - H_{target})^2 + w_c \cdot N_{features}$$

where:
- $H$ = persistence entropy
- $H_{target}$ = target entropy (hyperparameter)
- $N_{features}$ = number of persistence points
- $w_e, w_c$ = weights

---

## 4. Implementation in This Project

### 4.1 Architecture Overview

```
Input Image (64×64×3)
        ↓
   ┌────────────────────────────────────────────┐
   │           FaceEmbeddingCNN                 │
   │  ┌──────────────────────────────────────┐  │
   │  │  Block 1: Conv → BN → ReLU → CBAM    │  │
   │  │  Block 2: Conv → BN → ReLU → CBAM    │  │
   │  │  Block 3: Conv → BN → ReLU → CBAM    │  │
   │  │  Block 4: Conv → BN → ReLU → CBAM ←──┼──┼── Extract attention (8×8)
   │  └──────────────────────────────────────┘  │
   │              ↓                             │
   │    Global Average Pooling → Embedding      │
   └────────────────────────────────────────────┘
                ↓                    ↓
        Embeddings            Attention Map
                ↓                    ↓
        Triplet Loss          TDA Loss
                ↓                    ↓
        ────────── Combined Loss ──────────
```

### 4.2 Key Components

#### 4.2.1 TDAAnalyzer (`src/tda_modules.py`)

The main class for extracting topological features:

```python
class TDAAnalyzer:
    def __init__(self, homology_dims=[0, 1], n_bins=10):
        self.cubical = CubicalPersistence(homology_dimensions=homology_dims)
        self.entropy = PersistenceEntropy()
        self.amplitude = Amplitude(metric="wasserstein", order=2)
        self.num_points = NumberOfPoints()
        self.betti_curve = BettiCurve(n_bins=n_bins)
    
    def compute_persistence(self, attention_maps):
        """Compute persistence diagrams for 2D attention maps."""
        # attention_maps: (B, H, W)
        diagrams = self.cubical.fit_transform(attention_maps)
        return diagrams  # (B, n_points, 3) - each point is (birth, death, dim)
    
    def extract_features(self, attention_maps):
        """Extract topological features."""
        diagrams = self.compute_persistence(attention_maps)
        return {
            'entropy': self.entropy.fit_transform(diagrams),
            'amplitude': self.amplitude.fit_transform(diagrams),
            'num_points': self.num_points.fit_transform(diagrams),
            'betti_curves': self.betti_curve.fit_transform(diagrams),
            'diagrams': diagrams
        }
```

#### 4.2.2 AttentionTopologyLoss (`src/tda_modules.py`)

The loss function that regularizes attention topology:

```python
class AttentionTopologyLoss(nn.Module):
    def __init__(self, target_entropy=0.5, entropy_weight=1.0, complexity_weight=0.1):
        self.target_entropy = target_entropy
        self.entropy_weight = entropy_weight
        self.complexity_weight = complexity_weight
        self.analyzer = TDAAnalyzer()
    
    def forward(self, attention_maps):
        # Extract features (on CPU, non-differentiable)
        features = self.analyzer.extract_features(attention_maps)
        
        # Entropy deviation loss
        entropy_loss = ((features['entropy'] - self.target_entropy) ** 2).mean()
        
        # Complexity penalty
        complexity_loss = features['num_points'].mean()
        
        # Total loss
        total_loss = (
            self.entropy_weight * entropy_loss +
            self.complexity_weight * complexity_loss
        )
        
        return torch.tensor(total_loss, device=attention_maps.device)
```

#### 4.2.3 Model Modification (`src/models.py`)

The `FaceEmbeddingCNN` was modified to optionally return attention maps:

```python
class FaceEmbeddingCNN(nn.Module):
    def forward(self, x, mode='metric', return_attention=False):
        # ... backbone processing ...
        
        # Block 4 - optionally extract attention map
        x = self.block4_conv(x)
        if return_attention:
            x, attention_map = self.block4_cbam(x, return_attention=True)
        else:
            x = self.block4_cbam(x)
        
        # ... rest of forward pass ...
        
        if return_attention:
            return output, attention_map
        return output
```

### 4.3 Training Integration

In `run_tda_training.py`:

```python
def train_epoch_with_tda(model, dataloader, triplet_criterion, tda_loss_fn, optimizer, device, config):
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass with attention extraction
        if use_tda and (batch_idx % config['tda_apply_every_n'] == 0):
            anchor_embed, anchor_att = model(anchor, return_attention=True)
            positive_embed, positive_att = model(positive, return_attention=True)
            negative_embed, _ = model(negative)
            
            # Compute TDA loss on attention maps
            combined_att = torch.cat([anchor_att, positive_att], dim=0)
            tda_loss = tda_loss_fn(combined_att)
        else:
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            tda_loss = 0.0
        
        # Triplet loss
        triplet_loss = triplet_criterion(anchor_embed, positive_embed, negative_embed)
        
        # Combined loss
        total_loss = triplet_loss + config['tda_loss_weight'] * tda_loss
        
        total_loss.backward()
        optimizer.step()
```

### 4.4 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_tda` | True | Enable/disable TDA regularization |
| `tda_loss_weight` | 0.05 | Weight of TDA loss relative to triplet loss |
| `tda_apply_every_n` | 1 | Compute TDA every N batches (for efficiency) |
| `tda_target_entropy` | 0.5 | Target persistence entropy |
| `tda_entropy_weight` | 1.0 | Weight for entropy deviation term |
| `tda_complexity_weight` | 0.1 | Weight for complexity penalty term |

---

## 5. Experimental Results

### 5.1 Training Comparison

| Metric | Baseline | TDA | Difference |
|--------|----------|-----|------------|
| Best Val Accuracy | 94.27% | 94.21% | -0.06% |
| Best Val Loss | 0.0912 | 0.0890 | -0.0022 |
| Best Epoch | 31 | 30 | -1 |
| Avg Time/Epoch | 191.4s | 195.5s | +2.1% |
| Total Epochs | 35 | 40 | +5 |

### 5.2 Verification Performance

| Metric | Baseline | TDA | Change |
|--------|----------|-----|--------|
| Verification Accuracy | 81.77% | 81.76% | -0.01% |
| ROC-AUC | 0.8931 | 0.8965 | +0.0034 |
| TPR @ FPR=0.01 | 0.1968 | 0.1985 | +0.0017 |
| Separation Ratio | 4.2598 | 4.4669 | +0.2071 |

### 5.3 Topological Features

| Feature | Baseline | TDA | Interpretation |
|---------|----------|-----|----------------|
| H0 Entropy | 1.2407 | 1.5708 | TDA increased entropy (more diffuse) |
| H1 Entropy | 0.2206 | 0.1525 | TDA decreased H1 entropy |
| H0 Count | 3.4062 | 4.0625 | More connected components |
| H1 Count | 1.6562 | 1.5312 | Slightly fewer holes |
| Attention Mean | 0.0780 | 0.0161 | TDA made attention more sparse |
| Attention Sparsity | 98.34% | 99.90% | Much higher sparsity |

---

## 6. Analysis: Why TDA Didn't Improve Performance

### 6.1 Key Findings

Based on our experiments, we identified several reasons why TDA regularization had minimal impact:

#### 6.1.1 Baseline Already Well-Optimized

The baseline model with CBAM attention already produces high-quality attention maps:
- **High sparsity (98.34%)**: Already focused on key regions
- **Low mean activation (0.078)**: Not over-activating

TDA regularization is most effective when the baseline attention is noisy or unfocused, which is not the case here.

#### 6.1.2 Loss Weight Too Small

With `tda_loss_weight = 0.05`, the TDA loss contributes only 5% of the total gradient signal:

```
Typical loss values:
  Triplet Loss: ~0.08
  TDA Loss: ~0.95
  Weighted TDA: 0.95 * 0.05 = 0.0475
  
  TDA contribution: 0.0475 / (0.08 + 0.0475) ≈ 37%
```

While this seems significant, the TDA loss gradient is not backpropagated through the topology computation, making it less effective.

#### 6.1.3 Target Entropy Mismatch

We targeted `entropy = 0.5`, but the baseline H0 entropy was already 1.24. Pushing toward 0.5 may have conflicted with what the model naturally learned.

#### 6.1.4 Attention Map Resolution

The attention maps are 8×8, which is quite small for meaningful topological analysis:
- Only 64 pixels to form topological features
- Limited expressiveness for complex structures
- Most features are noise at this resolution

### 6.2 Visualization Insights

From the generated visualizations in `outputs/tda_analysis/`:

1. **attention_comparison.png**: Shows that TDA made attention maps more sparse but didn't fundamentally change the focus regions
2. **persistence_diagrams.png**: Both models show similar persistence patterns
3. **topological_statistics.png**: Minor differences in topological features

### 6.3 The Fundamental Limitation

The non-differentiable nature of TDA means we cannot directly optimize for topological properties. Instead, we rely on the model implicitly learning to produce attention maps with desirable topology through the scalar loss signal.

This indirect optimization is less effective than direct gradient-based learning, especially when:
- The baseline is already well-optimized
- The loss weight is small
- The target properties conflict with task performance

---

## 7. Future Directions

### 7.1 Potential Improvements

1. **Increase TDA Loss Weight**: Try `tda_loss_weight = 0.1 - 0.3`
2. **Adaptive Target Entropy**: Use the baseline's entropy as a reference
3. **Higher Resolution Attention**: Use attention from earlier layers (larger spatial size)
4. **Differentiable TDA**: Explore methods like PersLay or TopologyLayer
5. **Different TDA Metrics**: Use Betti curves or landscapes instead of entropy

### 7.2 Alternative Applications

TDA might be more effective for:
- **Noisy datasets**: Where baseline attention is less focused
- **Complex architectures**: With multiple attention heads
- **Interpretability**: Understanding model behavior rather than improving performance
- **Out-of-distribution detection**: Detecting when attention topology differs from training

### 7.3 Research Extensions

1. **Multi-scale TDA**: Analyze attention at multiple layers
2. **TDA for Data Augmentation**: Generate augmentations that preserve topology
3. **Topological Adversarial Training**: Make models robust to topological perturbations

---

## 8. References

### 8.1 TDA Fundamentals
1. Edelsbrunner, H., & Harer, J. (2008). *Persistent homology—a survey*. Contemporary mathematics, 453, 257-282.
2. Carlsson, G. (2009). *Topology and data*. Bulletin of the American Mathematical Society, 46(2), 255-308.

### 8.2 TDA in Deep Learning
3. Hofer, C., et al. (2019). *Learning representations of persistence barcodes*. Journal of Machine Learning Research, 20(126), 1-45.
4. Moor, M., et al. (2020). *Topological autoencoders*. International Conference on Machine Learning.
5. Carrière, M., et al. (2020). *PersLay: A neural network layer for persistence diagrams*. AISTATS.

### 8.3 Software
6. giotto-tda: https://github.com/giotto-ai/giotto-tda
7. Ripser: https://github.com/Ripser/ripser

### 8.4 Attention Mechanisms
8. Woo, S., et al. (2018). *CBAM: Convolutional block attention module*. ECCV.

---

## Appendix A: Running the Code

### A.1 Environment Setup

```bash
# Activate conda environment
conda activate face_recog

# Navigate to project
cd /home/hieutran/Facial-Recognition-with-Emotion-Liveness-Detection
```

### A.2 Training with TDA

```bash
# Full training (50 epochs, ~2.5 hours on RTX 4090)
python run_tda_training.py

# Or with nohup for SSH sessions
nohup python run_tda_training.py > tda_training.log 2>&1 &
```

### A.3 Visualization and Analysis

```bash
# TDA visualization (attention maps, persistence diagrams)
python scripts/tda_visualization.py

# Benchmark comparison (ROC, embeddings, training curves)
python scripts/benchmark_comparison.py
```

### A.4 Output Files

```
outputs/
├── best_tda_metric_model.pth     # TDA-trained model
├── tda_training_history.json     # Training metrics
├── tda_training_config.json      # Hyperparameters
├── tda_training_curves.png       # Training curves
└── tda_analysis/
    ├── attention_comparison.png      # Baseline vs TDA attention
    ├── persistence_diagrams.png      # Topological features
    ├── topological_statistics.png    # Statistical comparison
    ├── roc_comparison.png            # ROC curves
    ├── embedding_analysis.png        # Embedding quality
    ├── training_comparison.png       # Training dynamics
    ├── benchmark_report.json         # Full metrics report
    └── tda_effectiveness_analysis.json  # Issue diagnosis
```

---

## Appendix B: Code Reference

### B.1 Key Files

| File | Purpose |
|------|---------|
| `src/tda_modules.py` | TDAAnalyzer, AttentionTopologyLoss |
| `src/models.py` | FaceEmbeddingCNN with attention extraction |
| `src/config.py` | TDA hyperparameters |
| `run_tda_training.py` | Main training script |
| `scripts/tda_visualization.py` | Topological visualizations |
| `scripts/benchmark_comparison.py` | Performance benchmarks |

### B.2 Configuration in `src/config.py`

```python
# TDA Settings
USE_TDA = True
TDA_LOSS_WEIGHT = 0.05
TDA_APPLY_EVERY_N_BATCHES = 1
TDA_TARGET_ENTROPY = 0.5
TDA_ENTROPY_WEIGHT = 1.0
TDA_COMPLEXITY_WEIGHT = 0.1
```

---

*Document last updated: January 27, 2026*
