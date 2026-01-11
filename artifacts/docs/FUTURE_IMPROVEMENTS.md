# Future Improvement Steps

These are recommended optimizations to further improve MTL model performance, saved for future consideration.

## Priority Order

| Step | Action | Expected Gain | Time | Status |
|------|--------|---------------|------|--------|
| ✅ 1 | Add class weights + WeightedRandomSampler | +10-15% | 1-2h | **COMPLETED** |
| 2 | Unfreeze backbone (last 2 blocks) | +5-10% | 2-3h | Pending |
| 3 | Replace with real liveness data (NUAA) | +20-30% | 3-4h | Pending |
| 4 | Joint fine-tuning with all tasks | Polish | 4-6h | Pending |

---

## Step 2: Unfreeze Backbone (Last 2 Blocks)

### Rationale
Currently the backbone is frozen during task-specific training. Unfreezing the last 2 residual blocks allows the backbone to adapt its high-level features to the specific tasks while maintaining low-level feature stability.

### Implementation
```python
# In train_mtl.py, modify freeze_backbone logic:
def partial_unfreeze_backbone(model):
    """Unfreeze last 2 blocks of ResNet backbone."""
    # Freeze everything first
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Unfreeze layer3 and layer4 (last 2 blocks)
    for param in model.backbone.layer3.parameters():
        param.requires_grad = True
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
    
    # Also unfreeze attention modules if using CBAM
    if hasattr(model.backbone, 'cbam'):
        for param in model.backbone.cbam.parameters():
            param.requires_grad = True
```

### Training Considerations
- Use **lower learning rate** for backbone (1e-4) vs heads (1e-3)
- Add **gradient clipping** to prevent instability
- May need **more epochs** (70-100) for convergence

---

## Step 3: Real Liveness Data (NUAA Dataset)

### Rationale
Synthetic liveness data (even with improved augmentations) cannot fully capture real-world spoof patterns. The NUAA dataset contains real printed photo attacks captured in controlled conditions.

### Dataset Options

| Dataset | Size | Attack Types | Download |
|---------|------|--------------|----------|
| **NUAA** | ~1.5GB | Print attacks | [Link](http://parnec.nuaa.edu.cn/xtan/data/nuaa_imposter.html) |
| **CASIA-FASD** | ~2GB | Print, Replay, Cut | Request access |
| **Replay-Attack** | ~4GB | Print, Screen replay | [Idiap](https://www.idiap.ch/dataset/replayattack) |
| **CelebA-Spoof** | ~78GB | Multiple types | Kaggle (large!) |

### Implementation Steps
1. Download NUAA dataset (~1.5GB)
2. Extract and organize into `dataset/liveness/nuaa/train` and `test`
3. Update `data_loader_mtl.py` to support NUAA structure
4. Retrain liveness head with real data

### Expected Impact
- Accuracy: 62% → **80-85%**
- AUC: 0.67 → **0.90+**
- Better generalization to real-world spoofs

---

## Step 4: Joint Fine-tuning (All Tasks)

### Rationale
After training each task separately, joint fine-tuning allows the model to learn shared representations that benefit all tasks simultaneously.

### Implementation
```python
# Joint training configuration
JOINT_CONFIG = {
    'epochs': 30,
    'lr': 1e-4,  # Lower LR for fine-tuning
    'batch_size': 32,  # Smaller batch for memory
    'tasks': ['emotion', 'liveness'],
    'loss_weights': {
        'emotion': 1.0,
        'liveness': 1.0,
    },
    'use_uncertainty_weighting': True,  # Learn task weights
}
```

### Training Strategy
1. Load best checkpoints from emotion and liveness training
2. Combine both datasets using alternating batches or mixed sampling
3. Train with multi-task loss (uncertainty-weighted)
4. Monitor per-task metrics to ensure no degradation

### Expected Benefits
- Improved feature sharing between tasks
- More robust embeddings
- Better generalization

---

## Additional Considerations

### Computational Resources
- Steps 2-4 require more GPU memory and training time
- Consider using gradient accumulation for larger effective batch sizes
- Mixed precision training (AMP) helps with memory

### Evaluation Protocol
After each step, evaluate:
1. Per-task accuracy/metrics
2. Inference speed (should remain similar)
3. Model size (should remain ~50MB)

### Notes
- Complete steps in order for best results
- Each step builds on improvements from previous steps
- Monitor for overfitting when unfreezing backbone

---

*Last updated: January 11, 2026*
