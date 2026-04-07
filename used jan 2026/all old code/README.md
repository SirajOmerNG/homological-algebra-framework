# Homological Algebra Framework for Neural Network Interpretability

Implementation of the paper: **"A Homological Algebra Framework for Interpretable Deep Representations"**

## 📋 Overview

This framework models deep neural networks as **chain complexes** from homological algebra, enabling rigorous mathematical analysis of learned representations through topological invariants.

### Key Features

- ✅ **Chain Complex Modeling**: Neural layers as algebraic modules
- ✅ **Homology/Cohomology Computation**: Detect invariant features
- ✅ **Exact Sequence Analysis**: Characterize information flow
- ✅ **Bayesian Stability Analysis**: Assess topological robustness
- ✅ **Comprehensive Visualizations**: Publication-ready plots

## 🎯 What This Framework Does

### 1. **Topological Analysis**
- Computes **Betti numbers** (β_l) - measures representational complexity
- Identifies **cycles** and **boundaries** in representation spaces
- Detects **invariant features** that survive all transformations

### 2. **Information Flow Characterization**
- **Exact sequences** → Perfect information transmission
- **Non-exact sequences** → Information loss or redundancy
- Quantifies bottlenecks in network architecture

### 3. **Robustness Assessment**
- Perturbation analysis via Bayesian weight sampling
- Stability metrics for topological features
- Connection to generalization performance

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install torch torchvision numpy scipy matplotlib

# Clone/download the repository
# Files needed:
#   - homological_nn_framework.py
#   - example_usage.py
```

### Basic Usage

```python
from homological_nn_framework import HomologicalNN, HomologicalVisualizer
from example_usage import load_mnist
import torch

# Load data
train_loader, test_loader = load_mnist()

# Create network
model = HomologicalNN(
    layer_dims=[784, 128, 64, 32, 10],
    activation='relu'
)

# Train normally (standard PyTorch)
# ... training code ...

# Perform homological analysis
analysis = model.analyze_interpretability(
    test_loader,
    n_stability_samples=50
)

# Visualize results
HomologicalVisualizer.create_summary_report(analysis, save_dir='./results')
```

### Run Complete Experiments

```bash
python example_usage.py
```

This runs:
1. **MNIST experiment** - Simple digit classification
2. **CIFAR-10 experiment** - Natural image classification
3. **Comparative analysis** - Side-by-side comparison

## 📊 Datasets

### Experiment 1: MNIST
- **Size**: 70,000 images (60k train, 10k test)
- **Format**: 28×28 grayscale
- **Classes**: 10 digits (0-9)
- **Network**: [784 → 128 → 64 → 32 → 10]

**Expected Findings:**
- Early layers detect edge orientations (H¹)
- Middle layers capture digit topology (H²)
- Bottleneck at 32-dimensional layer
- High stability due to simple structure

### Experiment 2: CIFAR-10
- **Size**: 60,000 images (50k train, 10k test)
- **Format**: 32×32 RGB
- **Classes**: 10 objects (airplane, car, bird, etc.)
- **Network**: [3072 → 512 → 256 → 128 → 10]

**Expected Findings:**
- Higher Betti numbers (more complex representations)
- Multiple bottlenecks during compression
- Lower stability (natural image variability)
- Richer cohomological structure

## 📐 Mathematical Background

### Chain Complexes

A neural network is modeled as:
```
C: 0 → H⁽⁰⁾ --∂₁--> H⁽¹⁾ --∂₂--> ... --∂ₗ--> H⁽ᴸ⁾ → 0
```

Where:
- `H⁽ˡ⁾` = representation space at layer l
- `∂ₗ` = boundary operator (weight matrix + activation)
- Property: `∂ₗ₋₁ ∘ ∂ₗ = 0` (chain complex condition)

### Homology Groups

```
Hₗ(C) = ker(∂ₗ) / im(∂ₗ₊₁)
```

**Interpretation:**
- **Cycles** (ker ∂ₗ): Features killed by boundary operator
- **Boundaries** (im ∂ₗ₊₁): Features created from previous layer
- **Homology**: Intrinsic features (not from previous layer)

### Cohomology Groups

```
Hˡ(C) = ker(δˡ) / im(δˡ⁻¹)
```

Where `δˡ = (∂ₗ)ᵀ` (coboundary = transpose of boundary)

**Interpretation:**
- Dual perspective to homology
- Detects features that "survive" all subsequent operations
- Basis vectors = invariant feature detectors

### Betti Numbers

```
βₗ = rank(Hₗ) = dim(ker ∂ₗ) - dim(im ∂ₗ₊₁)
```

**Interpretation:**
- Measures "number" of independent topological features
- β₀: Connected components
- β₁: Holes/loops
- β₂: Voids/cavities
- Higher βₗ: More complex structure

### Exact Sequences

A sequence is **exact** at Cₗ if:
```
ker(∂ₗ) = im(∂ₗ₊₁)  ⟺  Hₗ(C) = 0
```

**Perfect information flow** (no loss, no redundancy)

### Information Loss

```
Lₗ = dim(ker ∂ₗ) - dim(im ∂ₗ₊₁)
```

- Lₗ > 0: Information loss (bottleneck)
- Lₗ < 0: Redundancy/expansion
- Lₗ = 0: Exact (perfect transmission)

### Stability Metric

```
Sₗ = 1 / Var(βₗ⁽ˢ⁾)
```

Where βₗ⁽ˢ⁾ are Betti numbers under weight perturbations

**High stability** → Robust topological features → Better generalization

## 📁 Output Files

After running experiments, you'll get:

```
results/
├── mnist/
│   ├── betti_numbers.png          # Topological complexity evolution
│   ├── exactness_analysis.png     # Information flow metrics
│   ├── stability_analysis.png     # Robustness under perturbations
│   └── training_curves.png        # Model performance
├── cifar10/
│   ├── betti_numbers.png
│   ├── exactness_analysis.png
│   ├── stability_analysis.png
│   └── training_curves.png
└── comparison.png                  # Side-by-side comparison
```

## 🔍 Interpreting Results

### Betti Numbers Plot
- **Y-axis**: Number of independent topological features
- **X-axis**: Layer depth
- **Interpretation**: 
  - Increasing β → Growing representational complexity
  - Decreasing β → Compression/simplification
  - Constant β → Information preserved

### Exactness Analysis
- **Y-axis**: Exactness measure [0, 1]
- **X-axis**: Layer depth
- **Red line**: Perfect exactness (1.0)
- **Interpretation**:
  - Close to 1.0 → Good information flow
  - Far from 1.0 → Bottleneck detected

### Information Loss Plot
- **Y-axis**: Lₗ (positive = loss, negative = expansion)
- **Zero line**: Perfect transmission
- **Interpretation**:
  - Large positive bars → Major bottlenecks
  - Negative bars → Feature expansion
  - Near zero → Efficient layer

### Stability Plot
- **Y-axis**: Stability score (log scale)
- **Higher is better**
- **Interpretation**:
  - High stability → Robust features
  - Low stability → Sensitive to perturbations
  - Correlates with generalization

## 🧪 Advanced Usage

### Custom Network Architecture

```python
model = HomologicalNN(
    layer_dims=[input_dim, 256, 128, 64, output_dim],
    activation='relu'  # or 'tanh', 'sigmoid'
)
```

### Manual Chain Complex Analysis

```python
from homological_nn_framework import ChainComplex, ExactSequenceAnalyzer

# Extract chain complex from trained model
chain_complex = model.extract_chain_complex(dataloader)

# Compute specific homology group
homology_basis, betti_number = chain_complex.compute_homology(layer_idx=2)

# Analyze exactness at specific layer
exactness = ExactSequenceAnalyzer.compute_exactness(chain_complex, layer_idx=2)

# Get information loss
info_loss = ExactSequenceAnalyzer.compute_information_loss(chain_complex, layer_idx=2)
```

### Custom Stability Analysis

```python
from homological_nn_framework import BayesianStabilityAnalyzer

analyzer = BayesianStabilityAnalyzer(noise_std=0.02)  # Custom noise level
stability_results = analyzer.compute_stability_metrics(
    chain_complex,
    n_samples=100  # More samples for better statistics
)
```

### Cohomology Feature Extraction

```python
# Get cohomology basis vectors (invariant features)
cohomology_basis, rank = chain_complex.compute_cohomology(layer_idx=2)

print(f"Found {rank} invariant features at layer 2")
print(f"Basis vectors shape: {cohomology_basis.shape}")

# These can be visualized as filters/patterns
```

## 📊 Key Theoretical Results

### Theorem 3.1 (Chain Complex Property)
For ReLU networks, ∂ₗ₋₁ ∘ ∂ₗ = 0 holds almost everywhere, validating the chain complex model.

### Theorem 3.5 (Invariant Feature Characterization)
Elements of Hˡ(C) correspond to features that:
1. Survive all subsequent boundary operations (invariance)
2. Are not generated by earlier layers (non-triviality)

### Theorem 3.7 (Stability-Generalization Connection)
Networks with higher cohomological stability tend to generalize better.

**Proof sketch**: Stable topological features → robust representations → tighter PAC-Bayes bounds

## 🎨 Visualization Examples

The framework generates publication-quality plots showing:

1. **Betti Number Evolution**
   - Tracks topological complexity through the network
   - Identifies where features emerge/disappear

2. **Exactness Analysis**
   - Shows where information flows perfectly
   - Highlights architectural bottlenecks

3. **Information Loss**
   - Quantifies compression at each layer
   - Reveals redundancy vs. loss

4. **Stability Metrics**
   - Robustness to weight perturbations
   - Predictive of generalization

## 🔬 Research Applications

This framework enables:

- ✅ **Architecture Design**: Identify optimal layer dimensions
- ✅ **Debugging**: Find where networks lose information
- ✅ **Feature Interpretation**: Extract semantically meaningful patterns
- ✅ **Robustness Analysis**: Predict generalization from topology
- ✅ **Pruning Guidance**: Remove redundant representations
- ✅ **Transfer Learning**: Identify which layers preserve information

## 📚 References

1. Original Paper: "A Homological Algebra Framework for Interpretable Deep Representations"
2. Hatcher, A. (2002). *Algebraic Topology*. Cambridge University Press.
3. Weibel, C. (1994). *An Introduction to Homological Algebra*. Cambridge University Press.

## 🐛 Troubleshooting

### Memory Issues
```python
# Reduce samples for analysis
analysis = model.analyze_interpretability(
    test_loader,
    n_stability_samples=20  # Reduce from 50
)

# Or reduce max_samples in chain complex extraction
chain_complex = model.extract_chain_complex(
    dataloader,
    max_samples=500  # Reduce from 1000
)
```

### Numerical Stability
```python
# Increase tolerance for kernel/image computation
chain_complex.compute_kernel(boundary_op, tolerance=1e-6)  # Default: 1e-8
```

### GPU Memory
```python
# Move model to CPU before analysis
model = model.cpu()
analysis = model.analyze_interpretability(test_loader)
```

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- Support for CNNs (spatial structure preservation)
- Persistent homology integration
- Sheaf cohomology for local-to-global analysis
- Directed homology for RNNs
- Categorical/functorial extensions

## 📄 License

MIT License - Feel free to use in research and applications

## 🙏 Acknowledgments

Based on the research paper introducing homological algebra to deep learning interpretability. This implementation provides a practical framework for researchers and practitioners.

---

**For questions or issues**: See the paper for theoretical details, or examine the code comments for implementation specifics.

**Citation**: If you use this framework, please cite the original paper.
