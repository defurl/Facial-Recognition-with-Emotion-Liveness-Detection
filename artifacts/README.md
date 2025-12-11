# Artifacts

Purpose: Training data, checkpoints, outputs, and scripts. Retrain here; export models for backend use.

Suggested contents
- dataset/ (raw and processed data)
- outputs/ (model checkpoints, embeddings, logs, thresholds)
- scripts/ (train_metric.py, train_softmax.py, eval scripts, migrations)
- analysis/ or notebooks if applicable

Workflow
1) Retrain models here; produce `outputs/best_metric_model.pth`, `employee_db.pt`, etc.
2) Backend reads artifacts via configured paths (e.g., ../artifacts/outputs/best_metric_model.pth).
3) Keep large data and generated files out of frontend/backend images; mount or sync as needed.
