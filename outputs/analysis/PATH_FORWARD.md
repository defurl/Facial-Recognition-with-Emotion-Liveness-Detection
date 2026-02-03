# Path Forward: TDA Face Recognition Improvement Plan

## Current State (Updated 2025-02-04)

### Completed Experiments Results

| Model | Val Acc | AUC | Epoch Peak | Status |
|-------|---------|-----|------------|--------|
| CNN Baseline | 94.15% | 93.49% | - | ✓ Baseline |
| TDA Regularized (λ=0.10) | 94.45% | 93.72% | - | ✓ Initial |
| **Exp A1 (λ=0.05)** | **94.54%** | **93.83%** | 6 | ✓ **Current Best** |
| Exp A2 (λ=0.02) | 94.42% | 93.73% | 2 | ✓ Complete |

### Key Findings from A1/A2 Analysis

1. **λ=0.05 is optimal so far** - Better than both λ=0.10 (too aggressive) and λ=0.02 (too weak)
2. **Early convergence** - A1 peaked at epoch 6, A2 peaked at epoch 2 (insufficient regularization)
3. **Overfitting after peak** - Both experiments showed declining val acc after their peaks
4. **AUC improvement correlates with λ** - Higher λ → better AUC (within reasonable range)

**Current improvement: +0.39% val acc, +0.34% AUC over CNN baseline**

---

## Experiments In Progress / Planned

### Exp A3: Fine-tuned Lambda (λ=0.03) - CREATED, NOT RUN
**Hypothesis**: Interpolate between A1 (0.05) and A2 (0.02) to find optimal λ

**Configuration**:
- TDA_LAMBDA = 0.03
- EARLY_STOPPING_PATIENCE = 12 (increased from 10)
- Uses existing intensity-based TDA cache

**Script**: `experiments/exp_a3_lambda_003/train.py`

---

### Exp B3: Edge-based TDA Features - CREATED, NOT RUN
**Hypothesis**: TDA on Canny edge maps captures facial structure better than raw intensity

**Configuration**:
- Canny edge detection (low=50, high=150) before TDA computation
- Same λ=0.05 as A1 for fair comparison
- Requires precomputation of edge TDA cache (~2h)

**Scripts**:
- Precompute: `experiments/exp_b3_edge_tda/precompute_edge_tda.py`
- Train: `experiments/exp_b3_edge_tda/train.py`

---

### Exp D: Ensemble Evaluation - CREATED, NOT RUN
**Hypothesis**: Combining CNN baseline + TDA-regularized embeddings improves verification

**Methods to test** (no training required):
1. Simple average: `(CNN + A1) / 2`
2. Weighted average: `α*CNN + (1-α)*A1` for α ∈ {0.3, 0.5, 0.7}
3. Concatenation: `[CNN || A1]` (512-dim combined)

**Script**: `experiments/exp_d_ensemble/evaluate.py`

---

## Experiment Execution Order

### Priority 1: Quick Wins (1-2 hours)
```bash
# Run Exp D first - no training, just inference
./experiments/run_experiments.sh d

# Run Exp A3 - uses existing TDA cache
./experiments/run_experiments.sh a3
```

### Priority 2: Feature Engineering (3-4 hours)
```bash
# Run Exp B3 - requires precomputation first
./experiments/run_experiments.sh b3
```

---

## Success Criteria Progress

| Target | Metric | Baseline | Current Best | Goal | Progress |
|--------|--------|----------|--------------|------|----------|
| Minimum | Val Acc | 94.15% | 94.54% | >94.5% | ✓ **ACHIEVED** |
| Good | Val Acc | 94.15% | 94.54% | >95.0% | 78% (0.46% to go) |
| Excellent | AUC | 93.49% | 93.83% | >94.0% | 67% (0.17% to go) |

---

## Analysis: Why λ=0.05 Works Best

From training history analysis:

| λ Value | Best Epoch | Final Loss | Peak Val Acc | Interpretation |
|---------|------------|------------|--------------|----------------|
| 0.10 | ~8 | 0.45 | 94.45% | TDA dominates, limits CNN learning |
| 0.05 | 6 | 0.40 | 94.54% | **Balanced regularization** |
| 0.02 | 2 | 0.35 | 94.42% | Too weak, peaks too early |

**Recommendation**: λ in range [0.03, 0.05] appears optimal for this architecture.

---

## Next Actions

1. **Immediate**: Run `./experiments/run_experiments.sh d` for ensemble evaluation
2. **Today**: Run `./experiments/run_experiments.sh a3` for λ=0.03 experiment
3. **Tomorrow**: Run `./experiments/run_experiments.sh b3` for edge-based TDA

---

## File Structure

```
experiments/
├── exp_a1_lambda_005/      # ✓ COMPLETE - Best model
│   ├── train.py
│   └── outputs/
│       ├── best_model.pth
│       └── training_history.json
├── exp_a2_lambda_002/      # ✓ COMPLETE
│   ├── train.py
│   └── outputs/
├── exp_a3_lambda_003/      # ○ CREATED - Not run
│   ├── train.py
│   └── outputs/
├── exp_b3_edge_tda/        # ○ CREATED - Requires precompute
│   ├── precompute_edge_tda.py
│   ├── train.py
│   └── outputs/
├── exp_d_ensemble/         # ○ CREATED - Not run
│   ├── evaluate.py
│   └── outputs/
├── run_experiments.sh      # Launcher script
└── benchmark_experiments.py
```

---

*Last updated: 2025-02-04*
*Best model: Exp A1 (λ=0.05) - 94.54% val acc, 93.83% AUC*
