"""
Evaluation script for Emotion Detection Model
Generates confusion matrix, classification report, and per-class metrics
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)
from tqdm import tqdm

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import DEVICE, OUTPUT_DIR
from emotion_model import EmotionCNN
from data_loader_emotion import create_emotion_dataloaders, EMOTION_LABELS

# Paths
EMOTION_MODEL_PATH = OUTPUT_DIR / "best_emotion_model.pth"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "emotion_confusion_matrix.png"
EVALUATION_RESULTS_PATH = OUTPUT_DIR / "emotion_evaluation_results.json"


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Returns:
        all_preds: Predicted labels
        all_labels: True labels
        all_probs: Prediction probabilities
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    
    # Normalized (percentages)
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")
    
    return cm, cm_normalized


def evaluate_emotion_model(model_path=None, batch_size=128):
    """
    Full evaluation pipeline for emotion model.
    
    Args:
        model_path: Path to trained model (default: best_emotion_model.pth)
        batch_size: Batch size for evaluation
    """
    if model_path is None:
        model_path = EMOTION_MODEL_PATH
    
    print("=" * 70)
    print("EMOTION MODEL EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    # Check model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first: python scripts/train_emotion.py")
        return None
    
    # Load data
    print("\n[1/4] Loading test data...")
    data_dict = create_emotion_dataloaders(batch_size=batch_size)
    test_loader = data_dict['test_loader']
    
    # Load model
    print("\n[2/4] Loading model...")
    model = EmotionCNN(num_classes=len(EMOTION_LABELS), use_cbam=True)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    
    # Evaluate
    print("\n[3/4] Evaluating on test set...")
    preds, labels, probs = evaluate_model(model, test_loader, DEVICE)
    
    # Compute metrics
    print("\n[4/4] Computing metrics...")
    
    # Overall metrics
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    
    # Per-class metrics
    emotion_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
    report = classification_report(
        labels, preds,
        target_names=emotion_names,
        output_dict=True
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1-Score: {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score: {f1_weighted * 100:.2f}%")
    print(f"Macro Precision: {precision * 100:.2f}%")
    print(f"Macro Recall: {recall * 100:.2f}%")
    
    print("\nPer-Class Results:")
    print("-" * 50)
    print(f"{'Emotion':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 50)
    
    for emotion in emotion_names:
        p = report[emotion]['precision'] * 100
        r = report[emotion]['recall'] * 100
        f1 = report[emotion]['f1-score'] * 100
        s = int(report[emotion]['support'])
        print(f"{emotion:<12} {p:>10.1f}% {r:>10.1f}% {f1:>10.1f}% {s:>10}")
    
    print("-" * 50)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm, cm_norm = plot_confusion_matrix(
        labels, preds, emotion_names, CONFUSION_MATRIX_PATH
    )
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'per_class': {
            emotion: {
                'precision': float(report[emotion]['precision']),
                'recall': float(report[emotion]['recall']),
                'f1-score': float(report[emotion]['f1-score']),
                'support': int(report[emotion]['support'])
            }
            for emotion in emotion_names
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open(EVALUATION_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {EVALUATION_RESULTS_PATH}")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Emotion Detection Model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    evaluate_emotion_model(
        model_path=args.model_path,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
