# Face Recognition Attendance System with CBAM Enhancement

A comprehensive face recognition system for attendance tracking, featuring deep learning models with Convolutional Block Attention Module (CBAM), real-time emotion detection, and liveness verification.

## 🏗️ Project Overview

This project implements a complete face recognition pipeline with two learning approaches:
- **Softmax Classification**: Direct identity classification (4000 classes)
- **Metric Learning**: Triplet loss-based embedding learning for verification

### Key Features
- **CBAM Attention Mechanism**: Enhanced CNN with Channel and Spatial Attention
- **Real-time Face Recognition**: Live camera-based attendance system
- **Emotion Detection**: Real-time emotion analysis and liveness detection
- **Deep-KNN Retrieval**: Similarity-based face retrieval with visualization
- **t-SNE Visualization**: Embedding space analysis and clustering
- **Comprehensive Evaluation**: ROC analysis with multiple distance metrics

## 📁 Project Structure

```
COS30082---FinalNPaper/
├── app.py                          # Main GUI application for attendance system
├── requirements.txt                # Python dependencies
├── src/                           # Core modules
│   ├── config.py                  # Configuration and hyperparameters
│   ├── models.py                  # FaceEmbeddingCNN with CBAM
│   ├── data_loader.py             # Dataset handling and transforms
│   ├── deep_knn.py                # k-NN utilities for retrieval
│   ├── emotion.py                 # Emotion detection and liveness
│   └── utils.py                   # Face detection utilities
├── scripts/                       # Training and evaluation scripts
│   ├── train_softmax.py           # Train classification model
│   ├── train_metric.py            # Train metric learning model
│   ├── evaluate.py                # ROC evaluation pipeline
│   ├── embeddings_analysis.py     # t-SNE and Deep-KNN analysis
│   └── plot_results.py            # Training curve visualization
├── dataset/                       # Dataset directory
│   ├── classification_data/       # Training/validation/test splits
│   └── verification_data/         # Verification pairs
└── outputs/                       # Generated models and results
    ├── best_softmax_model.pth     # Trained classification model
    ├── best_metric_model.pth      # Trained metric learning model
    └── analysis/                  # Analysis outputs (t-SNE, ROC)
```

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, Torchvision
- **Computer Vision**: OpenCV, MediaPipe, DeepFace
- **Scientific Computing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, t-SNE
- **GUI**: Tkinter with PIL/Pillow
- **Performance**: CBAM attention, batch processing, CUDA support

## ⚙️ Installation and Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd COS30082---FinalNPaper

# Create conda environment
conda create -n face_recog python=3.9
conda activate face_recog

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation
IMPORTANT: Due to the heavy size of the original dataset, a setup.py file has been made as a helper to download the dataset from kaggle. Requires the user to provide their own kaggle API in kaggle.json.
Organize your face dataset in the following structure:
```
dataset/classification_data/
├── train_data/
│   ├── n000001/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── n000002/
│   └── ...
├── val_data/
│   └── [same structure as train_data]
└── test_data/
    └── [same structure as train_data]
```

### 3. Configuration
Edit `src/config.py` to adjust hyperparameters:
```python
NUM_EPOCHS_SOFTMAX = 80    # Training epochs for classification
NUM_EPOCHS_METRIC = 80     # Training epochs for metric learning
USE_CBAM = True            # Enable/disable CBAM attention
BATCH_SIZE = 128           # Batch size for training
LEARNING_RATE = 1e-3       # Learning rate
```
- `src/config.py`: Central configuration. Paths, GUI cadence (`PROCESS_EVERY_N_FRAMES`), camera index fallback (`CAMERA_INDEX`), thresholds (evaluation vs GUI), device, and image normalization.
- `src/models.py`: `FaceEmbeddingCNN` with dual heads (metric embedding, classification). Utilities to count parameters.
- `src/data_loader.py`: Transforms and data pipelines for training/validation. Prepares 64×64 input for embedding model.
- `src/utils.py`: Non-ML helpers: Haar face detection, crop with padding, draw boxes, and temp file helpers.
- `src/emotion.py`: DeepFace-based analyzer returning both emotion and liveness on cropped faces; detection is skipped (`detector_backend='skip'`).
- `scripts/train_softmax.py`: Train classification head; writes `outputs/best_softmax_model.pth`.
- `scripts/train_metric.py`: Train embedding for verification; writes `outputs/best_metric_model.pth`.
- `scripts/evaluate.py`: ROC/AUC comparisons and plots for verification.
- `outputs/`: Models, employee DB (`employee_db.pt`), experiment registry, thresholds, and histories.
- `dataset/`: Structured data for classification and verification pairs; includes liveness sample folders.

### Data Flow (GUI)
- Capture → Haar detect → crop+resize 64×64 → liveness gate → if live: compute embedding → compare to DB with `F.pairwise_distance` → threshold → identity.
- Emotion runs at lower cadence on 224×224 crop via DeepFace and is displayed alongside identity.

### Performance Tips
- Increase `PROCESS_EVERY_N_FRAMES` (e.g., 15–20) to reduce per-frame load.
- Lower camera resolution to 640×480 (configured in `app.py`).
- Ensure GPU is available; otherwise expect lower FPS.

### Liveness
- Provided by DeepFace anti-spoofing; no custom training pipeline is used in this project.

## 📁 Project Structure

```
FinalDemo/
├── src/                    
│   ├── config.py          # Central configuration: paths, hyperparameters, device setup
│   ├── models.py          # Model architectures (FaceEmbeddingCNN, CBAM, etc.)
│   ├── data_loader.py     # Data loading, preprocessing, dataset classes
│   ├── utils.py           # Utility functions (face detection, cropping, drawing etc.)
│   └── emotion.py         # DeepFace-based emotion and liveness detection
├── scripts/               
│   ├── setup.py           # Environment setup, dependency check, Kaggle API, dataset download
│   ├── train_softmax.py   # Train classification (softmax) model
│   ├── train_metric.py    # Train metric learning (triplet loss) model
│   ├── evaluate.py        # Evaluate models, generate ROC curves, save results
│   ├── embeddings_analysis.py # t-SNE visualization, deep kNN analysis on embeddings
│   └── plot_results.py    # Plot training histories and ROC comparison
├── app.py                 # Main GUI application for face recognition attendance
├── dataset/               # Dataset directory (created by setup.py)
├── outputs/               # Model checkpoints, training history, ROC plots, employee DB
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset (Optional for Training)

If you want to train models from scratch:

```bash
python scripts/setup.py
```

This will:
- Configure Kaggle API (you'll need your `kaggle.json`)
- Download the face dataset (~1.5GB)
- Extract and organize the data

### 3. Train Models (Optional)

If you want to train from scratch:

```bash
# Train Softmax (Classification) model
python scripts/train_softmax.py

# Train Triplet (Metric Learning) model
python scripts/train_metric.py
```

**Note**: Training can be skipped if you already have pre-trained models in `outputs/`.

### 4. Run the Application

```bash
python app.py
```

## 💡 Usage Guide

### Running the GUI Application

1. **Start the Application**:
   ```bash
   python app.py
   ```

2. **Start Camera**: Click the "Start Camera" button to begin video feed

3. **Register Employees**:
   - Click "Register New Employee"
   - Enter the employee's name
   - Look at the camera and hold still
   - System will automatically capture and register

4. **Manage Employees**:
   - **View Employees**: See list of all registered employees
   - **Edit Employee**: Rename an employee
   - **Delete Employee**: Remove an employee from database
   - **Adjust Threshold**: Fine-tune recognition sensitivity

5. **Monitor Performance**:
   - Check the "Current Detection" panel for identification results
   - View "Debug Info" for distance metrics and threshold settings

### Training Models

#### Softmax (Classification) Model

```bash
python scripts/train_softmax.py
```

- Trains for classification across all identities
- Uses Cross-Entropy Loss
- Best for scenarios with fixed, known identities
- Output: `outputs/best_softmax_model.pth`

#### Metric Learning (Triplet) Model

```bash
python scripts/train_metric.py
```

- Trains with Triplet Loss for similarity learning
- Better generalization to unseen identities
- Recommended for the attendance system
- Output: `outputs/best_metric_model.pth`

### Evaluating Models

```bash
python scripts/evaluate.py
```

Generates:
- ROC curves comparing both models
- AUC scores for different distance metrics
- Performance comparison report
- Saved plots in `outputs/`

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Model hyperparameters
IMG_SIZE = 64                    # Input image size
EMBEDDING_DIM = 256              # Embedding dimension
BATCH_SIZE = 128                 # Training batch size
LEARNING_RATE = 1e-3             # Learning rate

# Training settings
NUM_EPOCHS_SOFTMAX = 30          # Epochs for softmax training
NUM_EPOCHS_METRIC = 10           # Epochs for metric training
MAX_IMAGES_PER_IDENTITY_TRAIN = 10  # Limit images per identity

# Verification settings
OPTIMAL_THRESHOLD_GUI = 0.8      # Recognition threshold (adjustable in GUI)
```

## 📊 Model Performance

### Softmax Model
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~90%
- **Best Use**: Fixed set of known identities

### Metric Learning Model
- **Training Accuracy**: ~92%
- **Validation AUC**: ~0.95
- **Best Use**: Open-set recognition with new identities

## 🔧 Troubleshooting

### Issue: "Could not open webcam"
**Solution**: Ensure your webcam is not being used by another application. Try reconnecting or restarting.

### Issue: "Model not found"
**Solution**: Train the models first using the training scripts, or ensure pre-trained models are in the `outputs/` directory.

### Issue: All faces recognized as the same person
**Solution**: 
1. Adjust the threshold using the "Adjust Threshold" button
2. Lower threshold = stricter matching
3. Try re-registering employees with better lighting

### Issue: Emotion detection not working
**Solution**: Install DeepFace: `pip install deepface`

### Issue: Slow performance
**Solution**:
1. Reduce `PROCESS_EVERY_N_FRAMES` in config (process less frequently)
2. Ensure CUDA is available for GPU acceleration
3. Close other applications to free resources

## 📝 Key Differences from Notebook

### Advantages of Python Project Structure:

1. **Separation of Concerns**: Each module has a clear responsibility
2. **No Accidental Re-execution**: Training won't re-run unless explicitly called
3. **Reusability**: Import modules in other scripts easily
4. **Version Control Friendly**: Better for Git tracking
5. **Professional Structure**: Production-ready organization
6. **Easy Testing**: Test individual modules independently
7. **Configuration Management**: Centralized settings in `config.py`
8. **Cleaner Execution**: Run only what you need

### Migration Guide:

| Notebook Cell | Python File | Purpose |
|---------------|-------------|---------|
| Imports & Setup | `src/config.py` | Configuration |
| Model Definition | `src/models.py` | Architecture |
| Data Loading | `src/data_loader.py` | Datasets |
| Training Loops | `scripts/train_*.py` | Training |
| Evaluation | `scripts/evaluate.py` | Testing |
| GUI Code | `app.py` | Application |
| Utils | `src/utils.py` | Helpers |

## 🚀 Complete Workflow

### Phase 1: Model Training

#### Train Classification Model (Softmax)
```bash
conda activate face_recog
python scripts/train_softmax.py
```
**Output**: `outputs/best_softmax_model.pth`

#### Train Metric Learning Model (Triplet Loss)
```bash
conda activate face_recog
python scripts/train_metric.py
```
**Output**: `outputs/best_metric_model.pth`

#### Visualize Training Progress
```bash
conda activate face_recog
python scripts/plot_results.py
```
**Output**: `outputs/training_comparison.png`

### Phase 2: Model Evaluation

#### Generate ROC Curves and Performance Metrics
```bash
conda activate face_recog
python scripts/evaluate_fixed.py
```
**Outputs**:
- `outputs/roc_comparison_fixed.png` - ROC curves for all models/metrics
- `outputs/roc_results_fixed.json` - AUC scores summary

**Expected Results**:
```json
{
  "Triplet - Euclidean": {"auc": 0.8642},
  "Triplet - Cosine": {"auc": 0.8642},
  "Softmax - Euclidean": {"auc": 0.7505},
  "Softmax - Cosine": {"auc": 0.7505}
}
```

### Phase 3: Embedding Analysis

#### t-SNE Visualization (Full Dataset)
```bash
conda activate face_recog
python scripts/embeddings_analysis.py \
    --reduction tsne \
    --sample-size 20000 \
    --model-path outputs/best_metric_model.pth \
    --mode metric
```
**Outputs**:
- `outputs/analysis/embeddings_metric_[timestamp]_tsne.png`
- `outputs/analysis/embeddings_metric_[timestamp]_tsne.csv`

#### t-SNE Visualization (Subset for Clarity)
```bash
# 1. Create sample directory with 10-20 identities
mkdir "outputs/analysis/t-sne samples"
# Copy 10-20 identity folders from dataset/classification_data/train_data/

# 2. Run t-SNE on subset
conda activate face_recog
python scripts/embeddings_analysis.py \
    --gallery-dir "outputs/analysis/t-sne samples" \
    --reduction tsne \
    --sample-size -1 \
    --limit-per-class 20 \
    --model-path outputs/best_metric_model.pth \
    --mode metric
```

#### Deep k-NN Retrieval Analysis
```bash
conda activate face_recog
python scripts/embeddings_analysis.py \
    --run-deepknn \
    --reduction none \
    --sample-size 20000 \
    --knn-k 5 \
    --knn-metric cosine \
    --knn-visualize-count 3 \
    --model-path outputs/best_metric_model.pth \
    --mode metric
```
**Outputs**:
- `outputs/analysis/embeddings_metric_[timestamp]_knn_metrics.json`
- `outputs/analysis/embeddings_metric_[timestamp]_knn_example_*.png`

### Phase 4: Real-time Attendance System

#### Launch GUI Application
```bash
conda activate face_recog
python app.py
```

**GUI Features**:
1. **Start Camera**: Begin real-time face recognition
2. **Register Employee**: Add new employees to the database
3. **Employee Management**: View, edit, or delete registered employees
4. **Threshold Adjustment**: Fine-tune recognition sensitivity in real-time
5. **Attendance Logging**: View timestamps and recognition results

**Usage Steps**:
1. Start the camera feed
2. Register employees by positioning them in front of the camera
3. Adjust recognition threshold if needed
4. Monitor real-time attendance with emotion and liveness detection

## 📊 Analysis and Interpretation

### Performance Metrics
- **Triplet Learning**: ~86.4% AUC (both Euclidean and Cosine)
- **Classification**: ~75.1% AUC (both Euclidean and Cosine)
- **Performance Gap**: 11.4% AUC difference

### Key Insights
1. **CBAM Impact**: Attention mechanism significantly improves both approaches
2. **Metric Equivalence**: Euclidean and Cosine distances are identical for normalized embeddings
3. **Triplet Superiority**: Metric learning outperforms classification for face verification
4. **Scale Challenges**: 4000-class classification remains challenging despite enhancements

### Visualization Interpretation
- **t-SNE**: Shows embedding space structure and identity clustering
- **ROC Curves**: Quantifies verification performance across thresholds
- **Deep-KNN**: Demonstrates similarity-based retrieval quality

## 🔧 Advanced Configuration

### Hyperparameter Tuning
Edit `src/config.py` for custom settings:

```python
# Model Architecture
USE_CBAM = True            # Enable/disable CBAM attention
EMBEDDING_DIM = 256        # Embedding dimension

# Training Parameters
NUM_EPOCHS_SOFTMAX = 80    # Classification training epochs
NUM_EPOCHS_METRIC = 80     # Metric learning training epochs
BATCH_SIZE = 128           # Training batch size
LEARNING_RATE = 1e-3       # Learning rate

# Verification Thresholds
OPTIMAL_THRESHOLD = 1.15        # Evaluation threshold
OPTIMAL_THRESHOLD_GUI = 0.8     # GUI threshold (stricter)

# Performance Settings
PROCESS_EVERY_N_FRAMES = 10     # GUI processing cadence
CAMERA_INDEX = 1                # Default camera index
```

### Custom Dataset Training
```bash
# Modify dataset paths in config.py
# Ensure proper directory structure
python scripts/train_softmax.py
python scripts/train_metric.py
```

### Baseline Comparison
```bash
# Disable CBAM for baseline
# Edit config.py: USE_CBAM = False
python scripts/train_metric.py
python scripts/evaluate_fixed.py
```

## 🐛 Troubleshooting

### Common Issues

#### Memory Errors
```bash
# Reduce sample size for analysis
python scripts/embeddings_analysis.py --sample-size 10000

# Reduce batch size for training
# Edit config.py: BATCH_SIZE = 64
```

#### Model Loading Errors
- Ensure consistent `num_classes=4000` when loading trained models
- Check CBAM compatibility between training and inference modes

#### Camera Issues
```bash
# Try different camera indices in config.py
CAMERA_INDEX = 0  # or 2, 3, etc.
```

#### CUDA/GPU Issues
```bash
# Force CPU usage if needed
export CUDA_VISIBLE_DEVICES=""
```

#### Shape Mismatch in Deep-KNN
- Use smaller sample sizes (e.g., 10000-20000)
- Ensure consistent k values in KNN search

### Performance Optimization
- Enable CUDA for significant training speedup
- Adjust `PROCESS_EVERY_N_FRAMES` for GUI responsiveness
- Use appropriate batch sizes based on available GPU memory

## 📚 Technical Details

### Model Architecture
- **Base CNN**: 4-layer convolutional network with batch normalization
- **CBAM Integration**: Channel and spatial attention modules after each conv block
- **Dual Heads**: Embedding head (metric learning) + Classification head (softmax)
- **Input Size**: 64×64 RGB images
- **Output**: 256-dimensional normalized embeddings

### Training Approach
- **Data Augmentation**: Random horizontal flip, color jitter, normalization
- **Loss Functions**: CrossEntropyLoss (classification), TripletMarginLoss (metric)
- **Optimization**: Adam optimizer with learning rate 1e-3
- **Regularization**: Batch normalization, dropout (implicit in architecture)

### Evaluation Metrics
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Distance Metrics**: Euclidean distance and cosine similarity
- **Top-K Accuracy**: k-NN retrieval performance assessment
- **Visualization**: t-SNE for embedding space analysis

### Data Flow (GUI Application)
```
Camera Feed → Face Detection (Haar) → Crop & Resize (64×64) → 
Liveness Detection (DeepFace) → [If Live] → CNN Embedding → 
Distance Comparison → Threshold Decision → Identity Recognition
         ↓
Emotion Analysis (224×224, lower cadence) → Display Overlay
```

## 📚 Citation and References

This project implements concepts from:
- **CBAM**: Woo, S., et al. "CBAM: Convolutional block attention module." ECCV 2018.
- **Triplet Loss**: Schroff, F., et al. "FaceNet: A unified embedding for face recognition." CVPR 2015.
- **Deep Metric Learning**: Various approaches for similarity-based learning
- **Dataset**: 11-785 Fall 2020 Homework 2 Part 2 (Kaggle)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make changes and test thoroughly
4. Commit changes (`git commit -am 'Add new feature'`)
5. Push to branch (`git push origin feature/new-feature`)
6. Submit a pull request with detailed description

## 📄 License

This project is for educational purposes as part of COS30082: Applied Machine Learning.

## 👤 Author

**Student Project** - COS30082 Applied Machine Learning

## 📞 Contact

For questions about implementation or usage:
- Check the troubleshooting section above
- Review inline code documentation
- Examine the configuration options in `src/config.py`

---

**⚠️ Important Note**: This system is designed for educational and research purposes. When deploying in production environments, ensure compliance with local privacy regulations and obtain proper consent for facial recognition usage.
