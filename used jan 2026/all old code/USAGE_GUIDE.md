# Complete Usage Instructions

## 📦 What You Have

You have 7 Python files for the Homological Algebra Framework:

1. **`download_datasets.py`** - Downloads MNIST & CIFAR-10
2. **`test_installation.py`** - Verifies everything works
3. **`homological_nn_framework.py`** - Core framework (main library)
4. **`example_usage.py`** - Complete experiments
5. **`run_all.py`** - Master script (runs everything)
6. **`requirements.txt`** - Dependencies list
7. **`README.md`** + **`QUICKSTART.md`** - Documentation

---

## 🎯 Three Ways to Run

### Method 1: Automatic (Easiest) ⭐

Run everything with one command:

```bash
python run_all.py
```

**Interactive menu will appear:**
```
RUN OPTIONS
What would you like to do?
  1. Complete workflow (download + test + experiments)  ← Choose this
  2. Download datasets only
  3. Test installation only
  4. Run experiments only (requires datasets)
  5. Run MNIST only (faster)
  6. Run CIFAR-10 only
  7. Show time estimates
  8. Exit
```

Just press `1` and follow prompts!

---

### Method 2: Step-by-Step (Recommended for Learning)

```bash
# Step 1: Download datasets (~3 minutes)
python download_datasets.py

# Step 2: Test installation (~30 seconds)
python test_installation.py

# Step 3: Run experiments (~20 minutes)
python example_usage.py
```

---

### Method 3: Individual Experiments (Custom)

**Just MNIST (faster):**
```python
from example_usage import experiment_mnist
model, results = experiment_mnist()
```

**Just CIFAR-10:**
```python
from example_usage import experiment_cifar10
model, results = experiment_cifar10()
```

---

## 📋 Detailed Instructions

### Before You Start

**Install dependencies:**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision numpy scipy matplotlib
```

**Check Python version:**
```bash
python --version  # Need 3.7 or higher
```

---

### Step 1: Download Datasets

**Option A: Automatic download**
```bash
python download_datasets.py
```

**What happens:**
1. Creates `./data/` directory
2. Downloads MNIST (11 MB)
   - 60,000 training images
   - 10,000 test images
   - 28×28 grayscale
3. Downloads CIFAR-10 (163 MB)
   - 50,000 training images
   - 10,000 test images
   - 32×32 RGB
4. Verifies downloads
5. (Optional) Shows sample images

**If datasets exist:**
The script will detect them and offer:
- Verify (check if working)
- Re-download (if corrupted)
- Show samples (visualize)

---

### Step 2: Test Installation

```bash
python test_installation.py
```

**This tests:**
- ✓ PyTorch/TorchVision installed
- ✓ NumPy/SciPy/Matplotlib installed
- ✓ Framework modules import correctly
- ✓ Chain complex operations work
- ✓ Neural network creation works
- ✓ Stability analyzer works
- ✓ CUDA available (if GPU present)

**Expected output:**
```
✅ ALL TESTS PASSED

Your installation is working correctly!
```

---

### Step 3: Run Experiments

**Full experiments (both datasets):**
```bash
python example_usage.py
```

**What happens:**
1. **MNIST Experiment** (~5-10 min)
   - Trains 4-layer network [784→128→64→32→10]
   - 10 epochs
   - Computes homology, cohomology, stability
   - Generates visualizations

2. **CIFAR-10 Experiment** (~10-15 min)
   - Trains 4-layer network [3072→512→256→128→10]
   - 15 epochs
   - Full homological analysis
   - Generates visualizations

3. **Comparative Analysis** (~1 min)
   - Side-by-side comparison
   - Complexity metrics
   - Stability comparison

**Output files created:**
```
results/
├── mnist/
│   ├── betti_numbers.png
│   ├── exactness_analysis.png
│   ├── stability_analysis.png
│   └── training_curves.png
├── cifar10/
│   ├── betti_numbers.png
│   ├── exactness_analysis.png
│   ├── stability_analysis.png
│   └── training_curves.png
└── comparison.png
```

---

## 🎨 Understanding the Output

### Terminal Output

You'll see detailed progress:

```
EXPERIMENT 1: MNIST DIGIT CLASSIFICATION
Network Architecture:
  Input:    784 (28×28 flattened)
  Hidden 1: 128
  Hidden 2: 64
  Hidden 3: 32 (bottleneck)
  Output:   10 (digit classes)

Training...
Epoch 1/10: Loss = 0.3245, Test Accuracy = 91.23%
...

HOMOLOGICAL ANALYSIS SUMMARY
[BETTI NUMBERS]
Layer 0: β_h =  2, β_c =  3
Layer 1: β_h =  1, β_c =  2
...

[EXACTNESS ANALYSIS]
Layer 0: Exactness = 0.987 ✓ Exact, Info Loss = 0
Layer 1: Exactness = 0.654 ✗ Non-exact, Info Loss = +5
...

⚠ Bottlenecks detected at layers: [2]
```

### Generated Plots

1. **`betti_numbers.png`** - Topological Complexity
   - X-axis: Layer depth
   - Y-axis: Number of features
   - Shows how features evolve through network

2. **`exactness_analysis.png`** - Information Flow
   - Left plot: Exactness score (higher = better)
   - Right plot: Information loss (bars)
   - Identifies bottlenecks

3. **`stability_analysis.png`** - Robustness
   - Left plot: Stability scores
   - Right plot: Variance of features
   - Predicts generalization

4. **`training_curves.png`** - Model Performance
   - Training loss over epochs
   - Test accuracy over time

5. **`comparison.png`** - MNIST vs CIFAR-10
   - 4 subplots comparing both datasets
   - Shows complexity differences

---

## ⚙️ Advanced Usage

### Custom Architecture

Edit `example_usage.py`:

```python
# Change network architecture
model = HomologicalNN(
    layer_dims=[input_size, 256, 128, 64, 32, output_size],
    activation='relu'  # or 'tanh', 'sigmoid'
)
```

### Reduce Analysis Time

```python
# Fewer stability samples (faster, less accurate)
results = model.analyze_interpretability(
    test_loader,
    n_stability_samples=20  # Default: 50
)

# Fewer training epochs
train_network(model, train_loader, test_loader, 
              epochs=5,  # Default: 10 for MNIST, 15 for CIFAR
              lr=0.001)
```

### Analyze Your Own Network

```python
from homological_nn_framework import HomologicalNN
from torch.utils.data import DataLoader

# Your trained PyTorch model
my_model = HomologicalNN(
    layer_dims=[your_dimensions],
    activation='relu'
)

# Load your trained weights
my_model.load_state_dict(torch.load('my_weights.pth'))

# Analyze!
results = my_model.analyze_interpretability(
    your_dataloader,
    n_stability_samples=50
)

# Visualize
from homological_nn_framework import HomologicalVisualizer
HomologicalVisualizer.create_summary_report(results, './my_results')
```

### Extract Specific Features

```python
# Get chain complex
chain = model.extract_chain_complex(dataloader)

# Compute homology at specific layer
homology_basis, betti = chain.compute_homology(layer_idx=2)
print(f"Layer 2 has {betti} invariant features")

# Get exact sequence metrics
from homological_nn_framework import ExactSequenceAnalyzer
exactness = ExactSequenceAnalyzer.compute_exactness(chain, layer_idx=2)
info_loss = ExactSequenceAnalyzer.compute_information_loss(chain, layer_idx=2)

print(f"Exactness: {exactness:.3f}")
print(f"Information loss: {info_loss}")
```

---

## 🔍 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

```bash
pip install torch torchvision
```

### "CUDA out of memory"

```python
# Use CPU instead
model = model.cpu()
results = model.analyze_interpretability(test_loader)
```

### "Download is too slow"

- Check internet connection
- Try later (servers might be busy)
- Download manually from official sources

### "Analysis takes forever"

```python
# Reduce samples
results = model.analyze_interpretability(
    test_loader,
    n_stability_samples=10  # Minimum for testing
)
```

### "Datasets already downloaded"

Run `download_datasets.py` and choose option 1 to verify.

### "Test installation failed"

1. Reinstall dependencies: `pip install --upgrade torch torchvision numpy scipy matplotlib`
2. Check Python version: `python --version` (need 3.7+)
3. Try in fresh environment: `python -m venv venv; source venv/bin/activate`

---

## 💡 Tips & Best Practices

### Performance Tips

1. **Use GPU if available** - 5-10x faster
2. **Start with MNIST** - Faster to debug
3. **Reduce epochs for testing** - 5 epochs is enough to test
4. **Lower stability samples** - 20 instead of 50 for quick tests

### Interpretation Tips

1. **High Betti numbers** → Complex representations
2. **Low exactness** → Information bottleneck
3. **Positive info loss** → Compression happening
4. **High stability** → Robust, generalizable features

### Development Tips

1. **Test on small data first** - Use 100 samples
2. **Save checkpoints** - Don't retrain if analysis fails
3. **Version control results** - Compare across experiments
4. **Document findings** - Keep notes on what works

---

## 📊 Expected Runtime

On CPU (Intel i7):
- Download datasets: 2-5 minutes
- MNIST training: 3-5 minutes
- MNIST analysis: 2-3 minutes
- CIFAR-10 training: 8-12 minutes
- CIFAR-10 analysis: 3-5 minutes
- **Total: ~20-30 minutes**

On GPU (NVIDIA RTX):
- MNIST: 1-2 minutes total
- CIFAR-10: 3-5 minutes total
- **Total: ~5-10 minutes**

---

## 📚 What to Read

1. **First time**: QUICKSTART.md
2. **Full details**: README.md
3. **Code understanding**: Comments in homological_nn_framework.py
4. **Theory**: Original research paper

---

## 🎯 Quick Reference Commands

```bash
# Complete automated workflow
python run_all.py

# Download datasets
python download_datasets.py

# Test installation
python test_installation.py

# Run all experiments
python example_usage.py

# MNIST only
python -c "from example_usage import experiment_mnist; experiment_mnist()"

# CIFAR-10 only
python -c "from example_usage import experiment_cifar10; experiment_cifar10()"
```

---

## ✅ Checklist

Before running experiments:
- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] ~200 MB free disk space
- [ ] Test installation passes (`python test_installation.py`)
- [ ] Datasets downloaded (auto-downloads on first run)

After experiments:
- [ ] Check `./results/` folder
- [ ] Review plots (betti_numbers, exactness, stability)
- [ ] Read terminal output summary
- [ ] Compare MNIST vs CIFAR-10 results

---

**Need help?** Check README.md for detailed documentation!
