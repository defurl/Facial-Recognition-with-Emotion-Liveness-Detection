# Path Forward: TDA Face Recognition Improvement Plan

## Current State (2026-02-03)

| Model | Val Acc | AUC | Status |
|-------|---------|-----|--------|
| CNN Baseline | 94.15% | 88.36% | ✓ Complete |
| Attention Fusion (TDA input) | 94.03% | - | ✗ Worse than baseline |
| **TDA Regularized** | **94.45%** | **89.46%** | ✓ **Current Best** |

**Improvement achieved: +0.30% val acc, +1.10% AUC**

---

## Priority Options for Further Improvement

### Option A: Optimize TDA Regularization Hyperparameters (High Priority)
**Goal**: Squeeze more from current approach

**Experiments**:
1. **Lower λ** (0.01-0.05) - Current 0.10 may be too aggressive
2. **No warmup** - Start TDA from epoch 1
3. **Cosine annealing for λ** - Dynamic λ that decreases over time
4. **Longer training** - 50 epochs with patience-based early stopping

**Expected Gain**: +0.1-0.3% val acc

**Effort**: Low (hyperparameter tuning only)

---

### Option B: TDA Feature Engineering (Medium Priority)
**Goal**: Create better TDA features that capture identity-relevant information

**Experiments**:
1. **Facial landmark-aligned TDA** 
   - Extract TDA only from eye, nose, mouth regions
   - These sub-regions have more identity-specific topology
   
2. **Multi-scale TDA**
   - Current: single 64x64 filtration
   - Try: 32x32, 64x64, 128x128 pyramid
   
3. **Edge-based TDA**
   - Run TDA on Canny edge maps instead of raw intensity
   - Edges capture facial structure better

**Expected Gain**: +0.2-0.5% val acc

**Effort**: Medium (requires TDA recomputation ~2h each)

---

### Option C: Alternative Regularization Losses (Medium Priority)
**Goal**: Better TDA loss formulation

**Experiments**:
1. **Contrastive TDA Loss** (instead of triplet)
   - Push different-person TDA features apart
   - Current loss only pulls same-person together
   
2. **KL Divergence TDA Loss**
   - Treat TDA as distribution, minimize divergence within identity
   
3. **Wasserstein Distance**
   - Natural metric for persistence diagrams
   - Theoretically grounded for TDA

**Expected Gain**: +0.1-0.4% val acc

**Effort**: Medium (loss function modifications)

---

### Option D: Ensemble Approach (Low Priority)
**Goal**: Combine CNN and TDA-regularized models

**Method**:
- Average embeddings: `final = α*CNN + (1-α)*TDA_reg`
- Or concatenate and train fusion layer

**Expected Gain**: +0.1-0.2% val acc

**Effort**: Low (inference-time fusion)

---

### Option E: Larger Backbone (Higher Effort)
**Goal**: Stronger base model for TDA to regularize

**Options**:
1. ResNet-50 backbone (current is ResNet-18 style)
2. EfficientNet-B2/B3
3. Vision Transformer (ViT-Small)

**Expected Gain**: +1-3% val acc

**Effort**: High (architecture change, full retraining)

---

## Recommended Path

### Phase 1: Quick Wins (1-2 days)
1. **A1**: Train with λ=0.05 instead of 0.10
2. **A2**: Train without warmup (TDA from epoch 1)
3. **D**: Try simple embedding averaging at inference

### Phase 2: Feature Engineering (3-5 days)
4. **B1**: Landmark-aligned TDA features
5. **B3**: Edge-based TDA

### Phase 3: Architecture (1 week+)
6. **E**: Larger backbone (if significant improvement needed)

---

## Success Criteria

| Target | Metric | Current | Goal |
|--------|--------|---------|------|
| Minimum | Val Acc | 94.45% | >94.5% |
| Good | Val Acc | 94.45% | >95.0% |
| Excellent | AUC | 89.46% | >90.0% |

---

## Next Action

**Immediate**: Run Option A1 - TDA regularization with λ=0.05

```bash
# Modify train_tda_regularized.py:
# tda_lambda = 0.05  (instead of 0.1)
# warmup_epochs = 0   (no warmup)
```

This is the lowest-effort experiment that could yield quick improvement.

---

*Plan created: 2026-02-03*
