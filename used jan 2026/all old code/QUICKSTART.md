# QUICK START GUIDE

## 🚀 Get Running in 4 Steps

### Step 1: Install Dependencies (2 minutes)

```bash
pip install torch torchvision numpy scipy matplotlib
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 2: Download Datasets (2-5 minutes)

```bash
python download_datasets.py
```

This downloads:
- MNIST: 11 MB (handwritten digits)
- CIFAR-10: 163 MB (natural images)
- Total: ~174 MB

**Or run everything automatically:**
```bash
python run_all.py
```

### Step 3: Test Installation (30 seconds)

```bash
python test_installation.py
```

You should see:
```
✅ ALL TESTS PASSED
```

### Step 4: Run Experiments (10-20 minutes)

**Option A: Run everything automatically (recommended)**
```bash
python run_all.py
# Interactive menu - choose option 1 for complete workflow
```

**Option B: Run step-by-step**
```bash
python download_datasets.py  # Download data first
python example_usage.py      # Then run experiments
```

**Option C: Just MNIST (faster, ~5 minutes)**
```python
from example_usage import experiment_mnist
model, results = experiment_mnist()
```

**Option C: Just CIFAR-10**
```python
from example_usage import experiment_cifar10
model, results = experiment_cifar10()
```

## 📊 What You'll Get

After running, check the `results/` folder:

```
results/
├── mnist/
│   ├── betti_numbers.png          ← Topological complexity
│   ├── exactness_analysis.png     ← Information flow
│   ├── stability_analysis.png     ← Robustness
│   └── training_curves.png        ← Model accuracy
├── cifar10/
│   └── (same files)
└── comparison.png                  ← MNIST vs CIFAR-10
```

## 🎯 Understanding the Results

### Betti Numbers (β_l)
- **High numbers** = Complex representations (many features)
- **Low numbers** = Simple/compressed representations
- **Changes** = Where feature abstraction happens

### Exactness
- **Close to 1.0** = Perfect information flow
- **Far from 1.0** = Bottleneck or information loss
- **Look for**: Drops indicate where network compresses

### Information Loss (L_l)
- **Positive bars** = Information loss (compression)
- **Negative bars** = Feature expansion
- **Zero** = Perfect transmission

### Stability (S_l)
- **Higher is better** = Robust features
- **Lower** = Sensitive to perturbations
- **Correlates with**: Generalization performance

## 💡 Minimal Example

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from homological_nn_framework import HomologicalNN

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128)

# Create network
model = HomologicalNN(
    layer_dims=[784, 128, 64, 10],
    activation='relu'
)

# Train your model here...
# ...

# Analyze!
results = model.analyze_interpretability(loader)

# Print summary
from homological_nn_framework import print_analysis_summary
print_analysis_summary(results)

# Visualize
from homological_nn_framework import HomologicalVisualizer
HomologicalVisualizer.create_summary_report(results, './my_results')
```

## ⚡ Quick Tips

1. **Start with MNIST** - it's faster and easier to interpret
2. **GPU recommended** - but CPU works fine (just slower)
3. **First run downloads data** - MNIST (11MB) + CIFAR-10 (163MB)
4. **Analysis takes time** - 50 stability samples × homology computations
5. **Reduce samples if slow** - Change `n_stability_samples=50` to `20`

## 🐛 Common Issues

### "ImportError: No module named torch"
```bash
pip install torch torchvision
```

### "CUDA out of memory"
```python
# Move model to CPU before analysis
model = model.cpu()
results = model.analyze_interpretability(loader)
```

### "Analysis takes too long"
```python
# Reduce samples
results = model.analyze_interpretability(
    loader,
    n_stability_samples=20  # Instead of 50
)
```

## 📖 Next Steps

- Read `README.md` for detailed documentation
- Check the paper for mathematical theory
- Modify `example_usage.py` for custom architectures
- Explore `homological_nn_framework.py` for advanced usage

## 🎓 Key Concepts in 30 Seconds

1. **Chain Complex**: Network = sequence of vector spaces + linear maps
2. **Homology**: Detects "holes" in representation space
3. **Betti Numbers**: Count independent topological features
4. **Exactness**: Measures perfect information transmission
5. **Stability**: Robustness of topology under perturbations

## 💬 Questions?

- Check the comprehensive README.md
- Read code comments in homological_nn_framework.py
- Refer to the original paper for theory

---

**Happy analyzing! 🎉**
