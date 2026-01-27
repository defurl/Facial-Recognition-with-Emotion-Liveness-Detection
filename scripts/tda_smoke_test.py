#!/usr/bin/env python3
"""
TDA Smoke Test Script

Tests TopologyLayer compatibility with PyTorch 2.7.1+CUDA and benchmarks overhead.
Run this BEFORE integrating TDA into the training pipeline.

Usage:
    python scripts/tda_smoke_test.py
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np

# ============= Environment Info =============
print("=" * 70)
print("TDA SMOKE TEST - Compatibility & Performance Check")
print("=" * 70)
print(f"\nPython: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print()

# ============= Test 1: TopologyLayer Installation =============
print("-" * 70)
print("TEST 1: TopologyLayer Import")
print("-" * 70)

try:
    from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, TopKBarcodeLengths
    print("✓ TopologyLayer imported successfully")
    TOPOLOGY_LAYER_AVAILABLE = True
except ImportError as e:
    print(f"✗ TopologyLayer import failed: {e}")
    print("\nTrying alternative: giotto-tda...")
    TOPOLOGY_LAYER_AVAILABLE = False

# ============= Test 2: Giotto-TDA Fallback =============
print("-" * 70)
print("TEST 2: Giotto-TDA Fallback")
print("-" * 70)

try:
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import PersistenceEntropy, BettiCurve
    print("✓ Giotto-TDA imported successfully")
    GIOTTO_AVAILABLE = True
except ImportError as e:
    print(f"✗ Giotto-TDA import failed: {e}")
    GIOTTO_AVAILABLE = False

# ============= Test 3: TopologyLayer Forward Pass =============
if TOPOLOGY_LAYER_AVAILABLE:
    print("-" * 70)
    print("TEST 3: TopologyLayer Forward Pass (4x4 tensor)")
    print("-" * 70)
    
    try:
        # Create a simple 4x4 attention map (mimics Block 4 CBAM output)
        attention_map = torch.randn(4, 4, requires_grad=True)
        
        # Initialize LevelSetLayer2D for 4x4 input, computing H0 and H1
        topo_layer = LevelSetLayer2D(size=(4, 4), sublevel=False)
        
        # Forward pass
        dgminfo = topo_layer(attention_map)
        print(f"✓ Forward pass successful")
        print(f"  Diagram info type: {type(dgminfo)}")
        print(f"  Diagram dimensions: {len(dgminfo)} (H0, H1, ...)")
        
        # Extract features
        sum_lengths = SumBarcodeLengths()
        features = sum_lengths(dgminfo)
        print(f"  Sum of barcode lengths: {features}")
        
        TOPO_FORWARD_OK = True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        TOPO_FORWARD_OK = False
else:
    TOPO_FORWARD_OK = False

# ============= Test 4: Gradient Flow =============
if TOPOLOGY_LAYER_AVAILABLE and TOPO_FORWARD_OK:
    print("-" * 70)
    print("TEST 4: Gradient Flow Through TopologyLayer")
    print("-" * 70)
    
    try:
        # Create input with gradient tracking
        attention_map = torch.randn(4, 4, requires_grad=True)
        
        # Forward
        topo_layer = LevelSetLayer2D(size=(4, 4), sublevel=False)
        dgminfo = topo_layer(attention_map)
        
        # Compute scalar loss from persistence
        sum_lengths = SumBarcodeLengths()
        loss = sum_lengths(dgminfo).sum()
        
        # Backward
        loss.backward()
        
        if attention_map.grad is not None:
            print(f"✓ Gradients computed successfully")
            print(f"  Gradient shape: {attention_map.grad.shape}")
            print(f"  Gradient norm: {attention_map.grad.norm().item():.6f}")
            GRADIENT_OK = True
        else:
            print("✗ Gradients are None")
            GRADIENT_OK = False
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        GRADIENT_OK = False
else:
    GRADIENT_OK = False

# ============= Test 5: Batch Processing Simulation =============
if TOPOLOGY_LAYER_AVAILABLE and TOPO_FORWARD_OK:
    print("-" * 70)
    print("TEST 5: Batch Processing (simulating training)")
    print("-" * 70)
    
    try:
        batch_size = 32
        spatial_size = (4, 4)
        
        # Simulate batch of attention maps
        batch_attention = torch.randn(batch_size, 1, *spatial_size, requires_grad=True)
        
        topo_layer = LevelSetLayer2D(size=spatial_size, sublevel=False)
        sum_lengths = SumBarcodeLengths()
        
        start_time = time.time()
        
        total_loss = 0.0
        for i in range(batch_size):
            dgminfo = topo_layer(batch_attention[i, 0])
            sample_loss = sum_lengths(dgminfo).sum()
            total_loss = total_loss + sample_loss
        
        # Backward through batch
        total_loss.backward()
        
        elapsed = time.time() - start_time
        per_sample_ms = (elapsed / batch_size) * 1000
        
        print(f"✓ Batch processing successful")
        print(f"  Batch size: {batch_size}")
        print(f"  Total time: {elapsed*1000:.2f} ms")
        print(f"  Per-sample time: {per_sample_ms:.2f} ms")
        print(f"  Gradient norm: {batch_attention.grad.norm().item():.6f}")
        
        BATCH_OK = True
        BATCH_TIME_MS = per_sample_ms
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        BATCH_OK = False
        BATCH_TIME_MS = float('inf')
else:
    BATCH_OK = False
    BATCH_TIME_MS = float('inf')

# ============= Test 6: Giotto-TDA Cubical Persistence =============
if GIOTTO_AVAILABLE:
    print("-" * 70)
    print("TEST 6: Giotto-TDA Cubical Persistence (4x4 grid)")
    print("-" * 70)
    
    try:
        # Create sample 4x4 attention map as numpy
        attention_np = np.random.randn(1, 4, 4).astype(np.float32)
        
        # Compute cubical persistence
        cp = CubicalPersistence(homology_dimensions=[0, 1])
        
        start_time = time.time()
        diagrams = cp.fit_transform(attention_np)
        elapsed = time.time() - start_time
        
        print(f"✓ Cubical persistence computed")
        print(f"  Diagram shape: {diagrams.shape}")
        print(f"  Time: {elapsed*1000:.2f} ms")
        
        # Extract features
        pe = PersistenceEntropy()
        entropy = pe.fit_transform(diagrams)
        print(f"  Persistence entropy: {entropy}")
        
        GIOTTO_OK = True
        GIOTTO_TIME_MS = elapsed * 1000
    except Exception as e:
        print(f"✗ Giotto-TDA computation failed: {e}")
        import traceback
        traceback.print_exc()
        GIOTTO_OK = False
        GIOTTO_TIME_MS = float('inf')
else:
    GIOTTO_OK = False
    GIOTTO_TIME_MS = float('inf')

# ============= Summary =============
print("\n" + "=" * 70)
print("SMOKE TEST SUMMARY")
print("=" * 70)

results = [
    ("TopologyLayer Import", TOPOLOGY_LAYER_AVAILABLE),
    ("Giotto-TDA Import", GIOTTO_AVAILABLE),
    ("TopologyLayer Forward", TOPO_FORWARD_OK if TOPOLOGY_LAYER_AVAILABLE else None),
    ("Gradient Flow", GRADIENT_OK if TOPOLOGY_LAYER_AVAILABLE else None),
    ("Batch Processing", BATCH_OK if TOPOLOGY_LAYER_AVAILABLE else None),
    ("Giotto-TDA Computation", GIOTTO_OK if GIOTTO_AVAILABLE else None),
]

for name, status in results:
    if status is True:
        print(f"  ✓ {name}: PASSED")
    elif status is False:
        print(f"  ✗ {name}: FAILED")
    else:
        print(f"  - {name}: SKIPPED")

print()

# Performance summary
if BATCH_OK:
    print(f"TopologyLayer overhead: {BATCH_TIME_MS:.2f} ms/sample")
    if BATCH_TIME_MS < 1.0:
        print("  → Excellent! Minimal impact on training.")
    elif BATCH_TIME_MS < 5.0:
        print("  → Good. Consider applying every N batches if needed.")
    elif BATCH_TIME_MS < 20.0:
        print("  → Moderate. Apply every 5-10 batches to limit overhead.")
    else:
        print("  → High overhead. Consider offline analysis only.")

if GIOTTO_OK:
    print(f"Giotto-TDA overhead: {GIOTTO_TIME_MS:.2f} ms/sample")

# Recommendation
print("\n" + "-" * 70)
print("RECOMMENDATION")
print("-" * 70)

if GRADIENT_OK and BATCH_OK and BATCH_TIME_MS < 10.0:
    print("✓ TopologyLayer is compatible and efficient.")
    print("  Proceed with differentiable TDA integration.")
    print("  Use LevelSetLayer2D on 4x4 CBAM attention maps.")
    recommendation = "topologylayer"
elif GIOTTO_OK:
    print("⚠ TopologyLayer not suitable for training integration.")
    print("  Use Giotto-TDA for offline/validation-time analysis.")
    print("  Compute persistence diagrams on attention maps post-hoc.")
    recommendation = "giotto-offline"
else:
    print("✗ No suitable TDA library available.")
    print("  Install TopologyLayer: pip install topologylayer")
    print("  Or install Giotto-TDA: pip install giotto-tda")
    recommendation = "install-required"

print(f"\nRecommendation: {recommendation}")
print("=" * 70)
