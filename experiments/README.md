# TDA Face Recognition Experiments

This folder contains organized experiments for TDA-based face recognition improvements.

## Structure

```
experiments/
├── run_experiments.sh          # Background training launcher
├── benchmark_experiments.py    # Benchmark all trained models
├── README.md                   # This file
│
├── exp_a1_lambda_005/          # Experiment A1: λ=0.05
│   ├── train.py                # Training script
│   └── outputs/
│       ├── best_model.pth      # Best model checkpoint
│       ├── training_history.json
│       ├── train.log           # Training log
│       └── nohup.out           # Nohup output
│
└── exp_a2_lambda_002/          # Experiment A2: λ=0.02
    ├── train.py                # Training script
    └── outputs/
        ├── best_model.pth
        ├── training_history.json
        ├── train.log
        └── nohup.out
```

## Experiments

### Experiment A1: λ=0.05 (Lower Lambda)
- **Hypothesis**: Lower TDA regularization weight allows better embedding quality
- **Configuration**:
  - TDA Lambda: 0.05 (vs 0.10 baseline)
  - Warmup: None (TDA from epoch 1)
  - Epochs: 30 with early stopping (patience 10)
  - Scheduler: Cosine Annealing with Warm Restarts

### Experiment A2: λ=0.02 (Very Low Lambda, Extended)
- **Hypothesis**: Minimal regularization with longer training
- **Configuration**:
  - TDA Lambda: 0.02
  - Warmup: None
  - Epochs: 50 with early stopping (patience 15)
  - Scheduler: Cosine Annealing with Warm Restarts

## Usage

### Launch Training (Background)

```bash
# Make executable (first time)
chmod +x experiments/run_experiments.sh

# Run all experiments
./experiments/run_experiments.sh all

# Run specific experiment
./experiments/run_experiments.sh a1
./experiments/run_experiments.sh a2

# Check status
./experiments/run_experiments.sh status
```

### Monitor Training

```bash
# Watch logs
tail -f experiments/exp_a1_lambda_005/outputs/train.log
tail -f experiments/exp_a2_lambda_002/outputs/train.log

# Watch GPU usage
watch -n 1 nvidia-smi

# Check running processes
ps aux | grep -E "exp_a[12]" | grep -v grep
```

### Stop Training

```bash
# Stop specific experiment
pkill -f "exp_a1_lambda_005/train.py"

# Stop all experiments
pkill -f "experiments/exp_a.*train.py"
```

### Benchmark Results

```bash
# After training completes
conda run -n face_recog python experiments/benchmark_experiments.py
```

## Baselines

| Model | Val Accuracy | Verification AUC |
|-------|-------------|------------------|
| CNN Baseline | 94.15% | 88.36% |
| TDA Regularized (λ=0.10) | 94.45% | 89.46% |
| Historical Best (concat) | 94.66% | - |

## Expected Results

Based on preliminary runs:
- **Exp A1 (λ=0.05)**: ~94.16% val acc, ~93.78% AUC (significantly improved AUC!)
- **Exp A2 (λ=0.02)**: TBD - hypothesis is further AUC improvement

## Notes

- Training uses pretrained CNN baseline weights as initialization
- TDA cache is shared from `outputs/tda_cache/tda_train.npz`
- Each experiment saves its own model, history, and logs
- Experiments can run concurrently if GPU memory allows
