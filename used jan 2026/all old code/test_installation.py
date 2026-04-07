"""
Quick Test Script - Verify Installation and Basic Functionality

This script performs a minimal test to ensure everything is working correctly.
Run this before running the full experiments.
"""

import sys
import numpy as np

print("="*70)
print("HOMOLOGICAL FRAMEWORK - INSTALLATION TEST")
print("="*70)

# Test 1: Import dependencies
print("\n[TEST 1] Checking dependencies...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError:
    print("  ✗ PyTorch not found. Install: pip install torch")
    sys.exit(1)

try:
    import torchvision
    print(f"  ✓ TorchVision {torchvision.__version__}")
except ImportError:
    print("  ✗ TorchVision not found. Install: pip install torchvision")
    sys.exit(1)

try:
    import scipy
    print(f"  ✓ SciPy {scipy.__version__}")
except ImportError:
    print("  ✗ SciPy not found. Install: pip install scipy")
    sys.exit(1)

try:
    import matplotlib
    print(f"  ✓ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("  ✗ Matplotlib not found. Install: pip install matplotlib")
    sys.exit(1)

# Test 2: Import framework
print("\n[TEST 2] Importing framework modules...")
try:
    from homological_nn_framework import (
        ChainComplex,
        ExactSequenceAnalyzer,
        BayesianStabilityAnalyzer,
        HomologicalNN,
        HomologicalVisualizer
    )
    print("  ✓ All framework modules imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import framework: {e}")
    sys.exit(1)

# Test 3: Basic chain complex operations
print("\n[TEST 3] Testing chain complex operations...")
try:
    # Create simple test data with proper dimensions
    np.random.seed(42)
    
    # Representation spaces (all 100 samples, different dimensions)
    H0 = np.random.randn(100, 10)  # 100 samples, 10 dimensions
    H1 = np.random.randn(100, 8)   # 100 samples, 8 dimensions
    H2 = np.random.randn(100, 5)   # 100 samples, 5 dimensions
    
    # Boundary operators (maps between consecutive spaces)
    # ∂_1: H0 (10D) → H1 (8D), so shape is (8, 10)
    boundary1 = np.random.randn(8, 10) * 0.1
    # ∂_2: H1 (8D) → H2 (5D), so shape is (5, 8)
    boundary2 = np.random.randn(5, 8) * 0.1
    
    # Create chain complex
    chain = ChainComplex(
        representation_spaces=[H0, H1, H2],
        boundary_operators=[boundary1, boundary2]
    )
    
    # Test homology computation
    homology_basis, betti = chain.compute_homology(0)
    print(f"  ✓ Homology computation: β₀ = {betti}")
    
    # Test cohomology computation
    cohomology_basis, betti_c = chain.compute_cohomology(0)
    print(f"  ✓ Cohomology computation: β⁰ = {betti_c}")
    
    # Test exactness
    exactness = ExactSequenceAnalyzer.compute_exactness(chain, 0)
    print(f"  ✓ Exactness analysis: {exactness:.3f}")
    
except Exception as e:
    print(f"  ✗ Chain complex operations failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Neural network creation
print("\n[TEST 4] Testing neural network creation...")
try:
    import torch.nn as nn
    
    # Create small network
    model = HomologicalNN(
        layer_dims=[10, 8, 4, 2],
        activation='relu'
    )
    
    # Test forward pass
    x = torch.randn(5, 10)
    y = model(x)
    
    assert y.shape == (5, 2), "Wrong output shape"
    print(f"  ✓ Network created: [10→8→4→2]")
    print(f"  ✓ Forward pass successful: {x.shape} → {y.shape}")
    
except Exception as e:
    print(f"  ✗ Neural network test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Stability analyzer
print("\n[TEST 5] Testing Bayesian stability analyzer...")
try:
    analyzer = BayesianStabilityAnalyzer(noise_std=0.01)
    
    # Test weight perturbation
    test_weight = np.random.randn(5, 10)
    perturbed = analyzer.perturb_weights(test_weight)
    
    assert perturbed.shape == test_weight.shape, "Wrong perturbed shape"
    diff = np.abs(perturbed - test_weight).mean()
    print(f"  ✓ Weight perturbation working (mean diff: {diff:.6f})")
    
    # Test posterior sampling
    boundaries = [np.random.randn(5, 10), np.random.randn(2, 5)]
    samples = analyzer.sample_posterior_weights(boundaries, n_samples=5)
    
    assert len(samples) == 5, "Wrong number of samples"
    print(f"  ✓ Posterior sampling: {len(samples)} samples generated")
    
except Exception as e:
    print(f"  ✗ Stability analyzer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check CUDA availability
print("\n[TEST 6] Checking compute device...")
if torch.cuda.is_available():
    print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠ CUDA not available - will use CPU (slower)")
    print("    This is fine, but experiments will take longer")

# Final summary
print("\n" + "="*70)
print("✅ ALL TESTS PASSED")
print("="*70)
print("\nYour installation is working correctly!")
print("\nNext steps:")
print("  1. Run full experiments: python example_usage.py")
print("  2. Or start with MNIST only (faster): see example_usage.py")
print("  3. Check README.md for detailed documentation")
print("\nNote: First run will download datasets (~170MB total)")
print("="*70)
