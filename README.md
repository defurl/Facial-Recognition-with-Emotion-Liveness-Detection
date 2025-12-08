# Face Recognition Attendance System with Liveness Detection

A comprehensive real-time face recognition system for attendance tracking with advanced liveness detection, emotion analysis, and explainable AI features.

## 🌟 Key Features

- **Real-time Face Recognition**: Live camera-based attendance with metric learning (triplet loss)
- **Advanced Liveness Detection**: Multi-method anti-spoofing (texture, color, motion, temporal analysis)
- **Emotion Detection**: Real-time emotion analysis using DeepFace
- **Explainable AI (XAI)**: Grad-CAM++ visualization for model interpretability
- **CBAM Attention**: Channel and Spatial Attention Module for enhanced feature extraction
- **Employee Management**: Easy registration, editing, and deletion of employees
- **Comprehensive Testing**: Full test suite for system validation

## 📁 Project Structure

```
Facial-Recognition-with-Emotion-Liveness-Detection/
├── app.py                          # Main GUI application
├── train_liveness.py               # CNN-based liveness detector training
├── demo_explainability.py          # XAI demonstration (Grad-CAM++)
├── test_all_modules.py             # Comprehensive system test
├── liveness_config.md              # Liveness detection configuration guide
├── requirements.txt                # Python dependencies
├── src/                            # Core modules
│   ├── config.py                   # Configuration and hyperparameters
│   ├── models.py                   # FaceEmbeddingCNN with CBAM
│   ├── data_loader.py              # Dataset handling and transforms
│   ├── liveness.py                 # Traditional CV-based liveness detection
│   ├── liveness_cnn.py             # CNN-based liveness detection
│   ├── emotion.py                  # Emotion & liveness integration
│   ├── explainability.py           # XAI utilities (Grad-CAM++)
│   ├── deep_knn.py                 # k-NN utilities for retrieval
│   ├── attendance.py               # Attendance logging
│   └── utils.py                    # Face detection utilities
├── scripts/                        # Training and evaluation
│   ├── setup.py                    # Dataset download (Kaggle)
│   ├── train_softmax.py            # Classification model training
│   ├── train_metric.py             # Metric learning training
│   ├── evaluate.py                 # ROC evaluation
│   ├── embeddings_analysis.py      # t-SNE and Deep-KNN analysis
│   └── plot_results.py             # Training visualization
├── dataset/                        # Face dataset
│   ├── classification_data/        # Train/val/test splits
│   └── verification_data/          # Verification pairs
└── outputs/                        # Models and results
    ├── best_metric_model.pth       # Trained metric learning model
    ├── employee_db.pt              # Employee embeddings database
    ├── attendance_log.csv          # Attendance records
    └── analysis/                   # Analysis outputs (t-SNE, ROC)
```

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd Facial-Recognition-with-Emotion-Liveness-Detection
```

### 2. Create Environment
```bash
conda create -n final-topic python=3.9
conda activate final-topic
pip install -r requirements.txt
```

### 3. Download Dataset (Optional - for training)
```bash
python scripts/setup.py
```
Requires Kaggle API (`kaggle.json`) to download the dataset (~1.5GB).

## 🚀 Quick Start

### Run the Attendance System
```bash
conda activate final-topic
python app.py
```

**Usage:**
1. **Start Camera** - Begin video feed
2. **Register Employee** - Add new employees by capturing their face
3. **Adjust Threshold** - Fine-tune recognition sensitivity (default: 0.8)
4. **View/Edit/Delete** - Manage registered employees
5. **Monitor** - Check attendance logs and detection metrics

### Test the System
```bash
conda activate final-topic
python test_all_modules.py
```

Validates all modules: face detection, emotion, liveness, embeddings, XAI.

## 🎓 Training (Optional)

### Train Metric Learning Model
```bash
python scripts/train_metric.py
```
- Uses triplet loss for embedding learning
- Output: `outputs/best_metric_model.pth`
- Best for open-set recognition

### Train Classification Model
```bash
python scripts/train_softmax.py
```
- Uses cross-entropy loss
- Output: `outputs/best_softmax_model.pth`
- Best for fixed identity sets

### Evaluate Models
```bash
python scripts/evaluate.py
```
Generates ROC curves and AUC scores comparing both approaches.

## 🔍 Advanced Features

### Liveness Detection

The system uses **gradual scoring** across 7 detection methods:

1. **Texture Analysis** (LBP) - 20% weight
2. **Color Distribution** (LAB) - 20% weight
3. **Moiré Patterns** (FFT) - 15% weight
4. **Motion Analysis** (Optical Flow) - 20% weight
5. **Edge Detection** (Hough Lines) - 8% weight
6. **Reflection Detection** - 7% weight
7. **Temporal Consistency** - 20% weight

**Threshold:** 60% confidence (configurable in `liveness_config.md`)

**Performance:**

For detailed configuration, see `liveness_config.md`.


## 🧩 Frame Pipeline & Hooks

The real-time loop is kept thin via `process_frame_shell` (see `src/pipeline/processing.py`).

- Detect callback: returns either `[(x, y, w, h), ...]` or a context dict with `faces` plus optional keys like `face_assignments`, `primary_face_id`, and `warnings`.
- Process callback: receives `(frame, face_idx, bbox, context)` and returns per-face results (identity, confidence, box color, etc.).
- UI callback: receives `(frame, results, context)` for drawing and status updates.

Hooks to extend:
- Liveness: `src/pipeline/liveness_adapter.py` wraps blink-only vs. heavier paths; integrate new signals there before the UI.
- Explainability: `ExplainabilityEngine` in `app.py` can consume the same per-face results.
- Async/optimized path: the vectorized pipeline currently lives in `src/performance_optimized_core.py`; keep callback shapes aligned with `process_frame_shell` when adding parity.
### Explainable AI (XAI)

```bash
python demo_explainability.py
```

Demonstrates Grad-CAM++ visualizations showing which facial regions the model focuses on for recognition decisions.

### Emotion Detection

Uses DeepFace for real-time emotion analysis:
- Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- Runs at lower cadence for performance (configurable)

## 📊 Model Performance

### Metric Learning (Triplet Loss)
- **AUC**: ~86.4%
- **Use Case**: Open-set recognition, new identities
- **Advantage**: Better generalization

### Classification (Softmax)
- **AUC**: ~75.1%
- **Use Case**: Fixed, known identities
- **Advantage**: Direct classification

### Liveness Detection
- **Traditional CV Methods**: 60-70% accuracy
- **CNN-based** (trainable): 90%+ expected
- **Real-time**: ✅ Gradual scoring prevents false positives

## ⚙️ Configuration

Edit `src/config.py`:

```python
# Model Settings
IMG_SIZE = 64                       # Input image size
EMBEDDING_DIM = 256                 # Embedding dimension
USE_CBAM = True                     # Enable attention module

# Recognition Settings
OPTIMAL_THRESHOLD_GUI = 0.8         # Recognition threshold
PROCESS_EVERY_N_FRAMES = 10         # Processing cadence

# Training Settings
BATCH_SIZE = 128                    # Training batch size
LEARNING_RATE = 1e-3                # Learning rate
NUM_EPOCHS_METRIC = 80              # Metric learning epochs
```

For liveness configuration, see `liveness_config.md`.

## 🔧 Troubleshooting

### Camera Issues
```bash
# Try different camera index in config.py
CAMERA_INDEX = 0  # or 1, 2
```

### Liveness Too Strict/Lenient
Adjust threshold in `liveness_config.md` or use GUI "Adjust Threshold" button.

### Slow Performance
1. Increase `PROCESS_EVERY_N_FRAMES` (e.g., 15-20)
2. Lower camera resolution to 640×480
3. Ensure GPU/CUDA available

### All Faces Recognized as Same Person
1. Lower threshold (stricter matching)
2. Re-register employees with better lighting
3. Check if employees are too similar

## 📈 Analysis Tools

### t-SNE Visualization
```bash
python scripts/embeddings_analysis.py \
    --reduction tsne \
    --sample-size 20000 \
    --model-path outputs/best_metric_model.pth
```

### Deep k-NN Retrieval
```bash
python scripts/embeddings_analysis.py \
    --run-deepknn \
    --knn-k 5 \
    --knn-visualize-count 3
```

### Training Curves
```bash
python scripts/plot_results.py
```

## 🛡️ Liveness Detection Research

The system includes both traditional CV and CNN-based liveness:

**Traditional CV** (current):
- 7 detection methods with gradual scoring
- Motion variance analysis
- 60-70% accuracy, real-time

**CNN-based** (trainable):
```bash
python train_liveness.py
```
- Expected 90%+ accuracy
- Requires labeled spoof dataset
- For research comparison

## 📚 Key Technologies

- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **MediaPipe**: Face detection
- **DeepFace**: Emotion analysis
- **Grad-CAM++**: Explainability
- **t-SNE**: Embedding visualization
- **Tkinter**: GUI framework

## 🎯 Project Highlights

1. **Dual Learning Approaches**: Softmax classification + Metric learning
2. **Multi-Method Liveness**: Traditional CV + CNN (trainable)
3. **Explainable AI**: Grad-CAM++ for model interpretability
4. **Production-Ready**: Real-time performance with robust error handling
5. **Research-Oriented**: Comprehensive evaluation and visualization tools

## 📝 File Descriptions

### Core Application
- `app.py`: Main GUI with face recognition, liveness, emotion, attendance
- `src/models.py`: FaceEmbeddingCNN with CBAM attention
- `src/liveness.py`: Traditional CV liveness (7 methods, gradual scoring)
- `src/emotion.py`: DeepFace integration for emotion + liveness

### Research Tools
- `demo_explainability.py`: Grad-CAM++ visualization demo
- `train_liveness.py`: CNN liveness detector training
- `scripts/embeddings_analysis.py`: t-SNE and Deep-KNN analysis
- `scripts/evaluate.py`: ROC curve generation

### Configuration
- `src/config.py`: Central configuration
- `liveness_config.md`: Comprehensive liveness tuning guide

## 🚨 Important Notes

1. **Dataset**: Use Kaggle API for download (see `scripts/setup.py`)
2. **Pre-trained Models**: Required for recognition (`outputs/best_metric_model.pth`)
3. **Liveness**: Default uses traditional CV; train CNN for better accuracy
4. **Privacy**: Obtain consent before deploying facial recognition
5. **Performance**: GPU recommended for real-time operation

## 📄 Citation

This project implements:
- **CBAM**: Woo, S., et al. "CBAM: Convolutional block attention module." ECCV 2018
- **FaceNet**: Schroff, F., et al. "FaceNet: A unified embedding for face recognition." CVPR 2015
- **Grad-CAM++**: Chattopadhay, A., et al. "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks." WACV 2018

## 👤 Author

**Academic Project** - COS30082: Applied Machine Learning

## 📞 Support

- Check `liveness_config.md` for liveness tuning
- Review `test_all_modules.py` for validation
- Examine inline code documentation
- Adjust settings in `src/config.py`

---

**⚠️ Disclaimer**: This system is for educational and research purposes. Ensure compliance with privacy regulations when deploying in production.
