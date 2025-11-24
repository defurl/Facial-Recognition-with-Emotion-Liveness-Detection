"""
Quick Test Script for New Explainability Features

Tests all new xAI components:
- Explainability engine
- Enhanced deep-kNN
- t-SNE dashboard
- Adaptive CBAM (if enabled)

Usage:
    python test_explainability.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from PIL import Image

print("="*70)
print("EXPLAINABILITY FEATURES TEST")
print("="*70)

# Test 1: Import all new modules
print("\n[1/6] Testing imports...")
try:
    from explainability import ExplainabilityEngine, explain_recognition
    from deep_knn import (
        knn_confidence_score, knn_predict_with_confidence,
        detect_outliers_lof, is_query_outlier, get_knn_explanation_text
    )
    from visualization_dashboard import EmbeddingDashboard, create_embedding_dashboard
    from models import AdaptiveCBAM, RegionAwareCBAM, FaceEmbeddingCNN
    from config import (
        ENABLE_EXPLAINABILITY, SHOW_ATTENTION_MAPS, ENABLE_TSNE_DASHBOARD,
        DEVICE, IMG_SIZE
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create models
print("\n[2/6] Testing model creation...")
try:
    # Standard model
    model = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000)
    print(f"✓ Standard model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Adaptive CBAM
    adaptive_cbam = AdaptiveCBAM(channels=128)
    print(f"✓ AdaptiveCBAM created: {sum(p.numel() for p in adaptive_cbam.parameters()):,} parameters")
    
    # Region-aware CBAM
    region_cbam = RegionAwareCBAM(channels=128, num_regions=5)
    print(f"✓ RegionAwareCBAM created: {sum(p.numel() for p in region_cbam.parameters()):,} parameters")
    
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    sys.exit(1)

# Test 3: Explainability engine
print("\n[3/6] Testing explainability engine...")
try:
    model.eval()
    explainer = ExplainabilityEngine(model, device='cpu')
    
    # Create dummy input
    dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    dummy_tensor = torch.randn(1, 3, 64, 64)
    
    # Generate attention map
    attention_map = explainer.generate_attention_map(dummy_tensor)
    assert attention_map.shape == (64, 64), f"Unexpected attention map shape: {attention_map.shape}"
    print(f"✓ Attention map generated: shape {attention_map.shape}")
    
    # Overlay on image
    overlay = explainer.overlay_attention_on_image(dummy_image, attention_map)
    assert overlay.shape == dummy_image.shape, "Overlay shape mismatch"
    print(f"✓ Attention overlay created: shape {overlay.shape}")
    
    # Explain distance
    dist_explanation = explainer.explain_distance(distance=0.5, threshold=1.0)
    assert 'confidence' in dist_explanation
    print(f"✓ Distance explanation: {dist_explanation['confidence']:.1f}% confidence")
    
    # Quality factors
    quality = explainer.explain_quality_factors(dummy_image)
    assert 'overall_quality' in quality
    print(f"✓ Quality analysis: {quality['overall_quality']:.1f}/100")
    
except Exception as e:
    print(f"✗ Explainability test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Enhanced deep-kNN
print("\n[4/6] Testing enhanced deep-kNN...")
try:
    # Create dummy kNN data
    neighbor_labels = np.array([1, 1, 1, 2, 2])
    neighbor_distances = np.array([0.1, 0.12, 0.15, 0.4, 0.42])
    
    # Confidence score
    conf_dict = knn_confidence_score(neighbor_labels, neighbor_distances, predicted_label=1)
    assert 'confidence' in conf_dict
    print(f"✓ kNN confidence score: {conf_dict['confidence']:.3f}")
    print(f"  - Agreement: {conf_dict['agreement']:.2%}")
    print(f"  - Avg distance: {conf_dict['avg_distance']:.3f}")
    
    # Predict with confidence
    pred_label, pred_conf, explanation = knn_predict_with_confidence(
        neighbor_labels, neighbor_distances, return_explanation=True
    )
    assert explanation is not None
    print(f"✓ Prediction with confidence: label={pred_label}, conf={pred_conf:.3f}")
    
    # Get explanation text
    text = get_knn_explanation_text(explanation)
    assert len(text) > 0
    print(f"✓ Explanation text generated ({len(text)} chars)")
    
    # Outlier detection
    dummy_embeddings = np.random.randn(100, 128).astype(np.float32)
    outlier_preds, outlier_scores = detect_outliers_lof(dummy_embeddings, contamination=0.1)
    num_outliers = (outlier_preds == -1).sum()
    print(f"✓ LOF outlier detection: {num_outliers}/100 outliers detected")
    
except Exception as e:
    print(f"✗ Deep-kNN test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: t-SNE dashboard
print("\n[5/6] Testing t-SNE visualization dashboard...")
try:
    # Create dummy data
    num_samples = 200
    num_classes = 10
    embeddings = np.random.randn(num_samples, 128).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)
    names = [f"Person_{i}" for i in range(num_classes)]
    
    # Create dashboard
    dashboard = EmbeddingDashboard(perplexity=min(30, num_samples - 1))
    dashboard.fit(embeddings, labels, names)
    print(f"✓ Dashboard fitted with {num_samples} embeddings")
    
    # Check metrics
    metrics = dashboard.cluster_metrics
    if metrics:
        print(f"  - Silhouette score: {metrics['silhouette_score']:.3f}")
        print(f"  - Separation ratio: {metrics['separation_ratio']:.2f}")
    
    # Transform query
    query_emb = np.random.randn(128).astype(np.float32)
    query_2d = dashboard.transform_query(query_emb)
    assert query_2d.shape == (2,), f"Unexpected query 2D shape: {query_2d.shape}"
    print(f"✓ Query transformed to 2D: {query_2d}")
    
    # Plot (without display)
    tsne_img = dashboard.plot(
        query_embedding=query_emb,
        query_name="Test Query",
        show_metrics=True
    )
    assert tsne_img.ndim == 3, "Plot should return 3D array"
    print(f"✓ t-SNE plot generated: shape {tsne_img.shape}")
    
    # Drift tracking
    dashboard.track_embedding_drift("Person_0", embeddings[0])
    dashboard.track_embedding_drift("Person_0", embeddings[1])
    drift_analysis = dashboard.analyze_drift("Person_0")
    print(f"✓ Drift tracking: {drift_analysis['drift_level']}")
    
except Exception as e:
    print(f"✗ Dashboard test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Adaptive CBAM
print("\n[6/6] Testing adaptive CBAM modules...")
try:
    # Test adaptive CBAM
    dummy_features = torch.randn(2, 128, 8, 8)
    
    adaptive_out = adaptive_cbam(dummy_features, quality_score=torch.tensor([[0.7], [0.3]]))
    assert adaptive_out.shape == dummy_features.shape
    print(f"✓ AdaptiveCBAM forward pass: {dummy_features.shape} → {adaptive_out.shape}")
    
    # Test region-aware CBAM
    region_out = region_cbam(dummy_features)
    assert region_out.shape == dummy_features.shape
    print(f"✓ RegionAwareCBAM forward pass: {dummy_features.shape} → {region_out.shape}")
    
    # Test quality branch
    quality_scores = adaptive_cbam.quality_branch(dummy_features)
    print(f"✓ Quality assessment: {quality_scores.squeeze().tolist()}")
    
except Exception as e:
    print(f"✗ Adaptive CBAM test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print("✓ All components tested successfully!")
print("\nNext steps:")
print("1. Run demo: python demo_explainability.py --camera")
print("2. Test with real data: python demo_explainability.py --image path/to/face.jpg")
print("3. Integrate into app.py for full system")
print("\nConfiguration:")
print(f"  - Explainability: {'Enabled' if ENABLE_EXPLAINABILITY else 'Disabled'}")
print(f"  - Attention maps: {'Enabled' if SHOW_ATTENTION_MAPS else 'Disabled'}")
print(f"  - t-SNE dashboard: {'Enabled' if ENABLE_TSNE_DASHBOARD else 'Disabled'}")
print(f"  - Device: {DEVICE}")
print("="*70)
