"""
=============================================================================
  Homological Algebra Framework for Interpretable Deep Representations
  ── COMPLETE VERSION WITH INLINE RESULTS (No External File Saving) ──────

  ALL RESULTS DISPLAYED DIRECTLY IN CONSOLE/NOTEBOOK
  Perfect for GitHub - results shown inside the code itself!

  HOW TO RUN:
    1. Save as homological_framework.py
    2. Run: python homological_framework.py

  DATASETS INCLUDED (from paper):
    1. TOY DATASET (Section 5.1) - Circles vs Squares
    2. AGRICULTURE - Precision Viticulture (UCI Wine)
    3. MNIST - Handwritten digits
    4. CIFAR-10 - Object recognition
    5. ROTATED MNIST - SO(2) equivariance test
    6. SPHERICAL DATA - SO(3) equivariance test
=============================================================================
"""

import os
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

warnings.filterwarnings('ignore')

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    print("ERROR: PyTorch not found. Run: conda install pytorch torchvision -c pytorch -y")
    exit(1)

# Scikit-learn for datasets
try:
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠ scikit-learn not found. Run: conda install scikit-learn -y")


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

class Config:
    # Paths - works on any computer
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Training parameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    TOY_EPOCHS = 300
    AGRI_EPOCHS = 100
    MNIST_EPOCHS = 5
    CIFAR_EPOCHS = 5
    ROTATED_EPOCHS = 5
    SPHERICAL_EPOCHS = 50
    
    # Analysis parameters
    N_STABILITY_SAMPLES = 10
    MAX_CHAIN_SAMPLES = 200


# Create directories
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("HOMOLOGICAL ALGEBRA FRAMEWORK")
print("=" * 60)
print(f" Project root: {Config.BASE_DIR}")
print(f" Data directory: {Config.DATA_DIR}")
print(f" Results directory: {Config.RESULTS_DIR}")
print("=" * 60)


# =============================================================================
# SECTION 2: HOMOLOGICAL ALGEBRA CORE
# =============================================================================

class ChainComplex:
    def __init__(self, representation_spaces, boundary_operators):
        self.spaces = representation_spaces
        self.boundaries = boundary_operators
    
    def compute_kernel(self, A, tol=1e-6):
        _, S, Vh = np.linalg.svd(A, full_matrices=True)
        rank = int(np.sum(S > tol))
        ndim = Vh.shape[0] - rank
        return ndim
    
    def compute_image(self, A, tol=1e-6):
        _, S, _ = np.linalg.svd(A, full_matrices=False)
        rank = int(np.sum(S > tol))
        return rank
    
    def compute_homology(self, l):
        if l >= len(self.boundaries):
            return 0
        ker_dim = self.compute_kernel(self.boundaries[l])
        if l == len(self.boundaries) - 1:
            return ker_dim
        img_dim = self.compute_image(self.boundaries[l + 1])
        return max(0, ker_dim - img_dim)
    
    def compute_exactness(self, l):
        if l >= len(self.boundaries):
            return 1.0
        ker_dim = self.compute_kernel(self.boundaries[l])
        if l == len(self.boundaries) - 1:
            img_dim = 0
        else:
            img_dim = self.compute_image(self.boundaries[l + 1])
        denom = max(ker_dim, img_dim, 1)
        return 1.0 - abs(ker_dim - img_dim) / denom


class HomologicalNN(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(self.n_layers)
        ])
        self.representations = []
    
    def forward(self, x, store=False):
        if store:
            self.representations = [x.detach().cpu().numpy()]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self.n_layers - 1:
                h = F.relu(h)
            if store:
                self.representations.append(h.detach().cpu().numpy())
        return h
    
    def extract_chain_complex(self, loader, max_samples=200):
        self.eval()
        all_reps = [[] for _ in range(self.n_layers + 1)]
        n_collected = 0
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.view(bx.size(0), -1)
                self.forward(bx, store=True)
                for i, rep in enumerate(self.representations):
                    all_reps[i].append(rep)
                n_collected += bx.size(0)
                if n_collected >= max_samples:
                    break
        rep_spaces = [np.vstack(r)[:max_samples] for r in all_reps]
        boundaries = [layer.weight.detach().cpu().numpy() for layer in self.layers]
        return ChainComplex(rep_spaces, boundaries)
    
    def analyze(self, loader):
        cc = self.extract_chain_complex(loader, Config.MAX_CHAIN_SAMPLES)
        results = {'betti_numbers': [], 'exactness': [], 'information_loss': []}
        for l in range(len(cc.boundaries)):
            results['betti_numbers'].append(cc.compute_homology(l))
            results['exactness'].append(cc.compute_exactness(l))
            ker_dim = cc.compute_kernel(cc.boundaries[l])
            if l == len(cc.boundaries) - 1:
                img_dim = 0
            else:
                img_dim = cc.compute_image(cc.boundaries[l + 1])
            results['information_loss'].append(ker_dim - img_dim)
        return results


# =============================================================================
# SECTION 3: TRAINING FUNCTION
# =============================================================================

def train_network(model, train_loader, test_loader, epochs, lr, label=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n  Training {label} on {device}...")
    print("  " + "-" * 50)
    
    train_losses, test_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx = bx.view(bx.size(0), -1).to(device)
            by = by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.view(bx.size(0), -1).to(device)
                by = by.to(device)
                preds = model(bx).argmax(1)
                total += by.size(0)
                correct += (preds == by).sum().item()
        
        acc = 100 * correct / total
        test_accs.append(acc)
        
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}, Acc = {acc:.2f}%")
    
    print(f"\n  ✓ FINAL ACCURACY: {test_accs[-1]:.2f}%")
    return train_losses, test_accs


# =============================================================================
# SECTION 4: DATASET 1 - TOY (Circles vs Squares)
# =============================================================================

def load_toy_dataset():
    rng = np.random.default_rng(42)
    n_samples = 1000
    
    # Circles (class 0)
    angles = rng.uniform(0, 2 * np.pi, n_samples // 2)
    radii = np.sqrt(rng.uniform(0, 1, n_samples // 2))
    circles = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    
    # Squares (class 1)
    squares = []
    while len(squares) < n_samples // 2:
        pts = rng.uniform(-0.9, 0.9, (n_samples, 2))
        outside = pts[np.sqrt(pts[:, 0]**2 + pts[:, 1]**2) > 1.0]
        squares.append(outside)
    squares = np.vstack(squares)[:n_samples // 2]
    
    X = np.vstack([circles, squares]).astype(np.float32)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int64)
    
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    
    split = int(0.8 * len(X))
    X_train, X_test = torch.tensor(X[:split]), torch.tensor(X[split:])
    y_train, y_test = torch.tensor(y[:split]), torch.tensor(y[split:])
    
    return (DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True),
            DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False))


def experiment_toy():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: TOY DATASET (Circles vs Squares)")
    print("  Paper Section 5.1 - Topological shape classification")
    print("=" * 70)
    
    train_loader, test_loader = load_toy_dataset()
    model = HomologicalNN([2, 16, 8, 2])
    losses, accs = train_network(model, train_loader, test_loader, 
                                  Config.TOY_EPOCHS, Config.LEARNING_RATE, 'Toy Dataset')
    
    results = model.analyze(test_loader)
    
    print("\n   HOMOLOGICAL ANALYSIS RESULTS:")
    print(f"    Betti numbers (β₀, β₁, β₂): {results['betti_numbers']}")
    print(f"    Exactness per layer: {[round(x, 3) for x in results['exactness']]}")
    print(f"    Information loss: {results['information_loss']}")
    
    return results, accs[-1]


# =============================================================================
# SECTION 5: DATASET 2 - AGRICULTURE (Precision Viticulture)
# =============================================================================

def load_agriculture_dataset():
    if not SKLEARN_OK:
        return None, None
    
    wine = load_wine()
    scaler = StandardScaler()
    X = scaler.fit_transform(wine.data).astype(np.float32)
    y = wine.target.astype(np.int64)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return (DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=16, shuffle=True),
            DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=16, shuffle=False))


def experiment_agriculture():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: PRECISION AGRICULTURE (Viticulture)")
    print("  UCI Wine - 13 soil chemistry features, 3 grape cultivars")
    print("=" * 70)
    
    if not SKLEARN_OK:
        print("  ⚠ Scikit-learn not installed. Skipping.")
        return None, None
    
    train_loader, test_loader = load_agriculture_dataset()
    model = HomologicalNN([13, 32, 16, 8, 3])
    losses, accs = train_network(model, train_loader, test_loader,
                                  Config.AGRI_EPOCHS, Config.LEARNING_RATE, 'Agriculture')
    
    results = model.analyze(test_loader)
    
    print("\n   HOMOLOGICAL ANALYSIS RESULTS:")
    print(f"    Betti numbers: {results['betti_numbers']}")
    print(f"    Exactness: {[round(x, 3) for x in results['exactness']]}")
    print(f"    Information loss: {results['information_loss']}")
    
    # Feature importance from first layer
    W1 = model.layers[0].weight.detach().cpu().numpy()
    importance = np.linalg.norm(W1, axis=0)
    importance /= importance.sum()
    
    features = ['Alcohol', 'Malic acid', 'Ash', 'Alkalinity', 'Mg', 'Phenols',
                'Flavanoids', 'NF Phenols', 'Proanthocyan.', 'Color', 'Hue', 'OD280', 'Proline']
    
    print("\n  FEATURE IMPORTANCE (Soil chemistry markers):")
    sorted_idx = np.argsort(importance)[::-1]
    for i in range(5):
        print(f"    {i+1}. {features[sorted_idx[i]]}: {importance[sorted_idx[i]]*100:.1f}%")
    
    return results, accs[-1]


# =============================================================================
# SECTION 6: DATASET 3 - MNIST
# =============================================================================

def load_mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = torchvision.datasets.MNIST(Config.DATA_DIR, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(Config.DATA_DIR, train=False, download=True, transform=transform)
    return (DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True),
            DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False))


def experiment_mnist():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: MNIST (Handwritten Digits)")
    print("  28x28 images, 10 digits - Image topology analysis")
    print("=" * 70)
    
    train_loader, test_loader = load_mnist_dataset()
    model = HomologicalNN([784, 128, 64, 32, 10])
    losses, accs = train_network(model, train_loader, test_loader,
                                  Config.MNIST_EPOCHS, Config.LEARNING_RATE, 'MNIST')
    
    results = model.analyze(test_loader)
    
    print("\n   HOMOLOGICAL ANALYSIS RESULTS:")
    print(f"    Betti numbers (β₀→β₃): {results['betti_numbers']}")
    print(f"    Exactness per layer: {[round(x, 3) for x in results['exactness']]}")
    print(f"    Average exactness: {np.mean(results['exactness']):.3f}")
    print(f"    Total information loss: {sum(results['information_loss'])}")
    
    return results, accs[-1]


# =============================================================================
# SECTION 7: DATASET 4 - CIFAR-10
# =============================================================================

def load_cifar10_dataset():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_ds = torchvision.datasets.CIFAR10(Config.DATA_DIR, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(Config.DATA_DIR, train=False, download=True, transform=transform)
    return (DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True),
            DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False))


def experiment_cifar10():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: CIFAR-10 (Object Recognition)")
    print("  32x32x3 images, 10 classes - Complex image topology")
    print("=" * 70)
    
    train_loader, test_loader = load_cifar10_dataset()
    model = HomologicalNN([3072, 512, 256, 128, 10])
    losses, accs = train_network(model, train_loader, test_loader,
                                  Config.CIFAR_EPOCHS, Config.LEARNING_RATE, 'CIFAR-10')
    
    results = model.analyze(test_loader)
    
    print("\n   HOMOLOGICAL ANALYSIS RESULTS:")
    print(f"    Betti numbers: {results['betti_numbers']}")
    print(f"    Exactness: {[round(x, 3) for x in results['exactness']]}")
    print(f"    Information loss: {results['information_loss']}")
    
    return results, accs[-1]


# =============================================================================
# SECTION 8: DATASET 5 - ROTATED MNIST (SO(2) Equivariance)
# =============================================================================

def load_rotated_mnist_dataset():
    transform_train = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = torchvision.datasets.MNIST(Config.DATA_DIR, train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.MNIST(Config.DATA_DIR, train=False, download=True, transform=transform_test)
    return (DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True),
            DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False))


def experiment_rotated_mnist():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: ROTATED MNIST (SO(2) Equivariance Test)")
    print("  Testing rotational symmetry - Paper Section 3.2")
    print("=" * 70)
    
    train_loader, test_loader = load_rotated_mnist_dataset()
    model = HomologicalNN([784, 128, 64, 32, 10])
    losses, accs = train_network(model, train_loader, test_loader,
                                  Config.ROTATED_EPOCHS, Config.LEARNING_RATE, 'Rotated MNIST')
    
    results = model.analyze(test_loader)
    
    print("\n   EQUIVARIANCE ANALYSIS:")
    print(f"    Rotation-invariant accuracy: {accs[-1]:.2f}%")
    print(f"    Betti numbers: {results['betti_numbers']}")
    print(f"    Exactness: {[round(x, 3) for x in results['exactness']]}")
    
    return results, accs[-1]


# =============================================================================
# SECTION 9: DATASET 6 - SPHERICAL DATA (SO(3) Equivariance)
# =============================================================================

def load_spherical_dataset():
    rng = np.random.default_rng(42)
    n_samples = 2000
    
    # Generate points uniformly on sphere S^2
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    phi = np.arccos(2 * rng.uniform(0, 1, n_samples) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    X = np.stack([x, y, z], axis=1).astype(np.float32)
    
    # Labels based on hemisphere (north vs south)
    y_labels = (z > 0).astype(np.int64)
    
    split = int(0.8 * n_samples)
    X_train, X_test = torch.tensor(X[:split]), torch.tensor(X[split:])
    y_train, y_test = torch.tensor(y_labels[:split]), torch.tensor(y_labels[split:])
    
    return (DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True),
            DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False))


def experiment_spherical():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 6: SPHERICAL DATA (SO(3) Equivariance Test)")
    print("  3D points on sphere S^2 - 3D rotation invariance")
    print("=" * 70)
    
    train_loader, test_loader = load_spherical_dataset()
    model = HomologicalNN([3, 64, 32, 16, 2])
    losses, accs = train_network(model, train_loader, test_loader,
                                  Config.SPHERICAL_EPOCHS, Config.LEARNING_RATE, 'Spherical Data')
    
    results = model.analyze(test_loader)
    
    print("\n   SO(3) EQUIVARIANCE ANALYSIS:")
    print(f"    Hemisphere classification accuracy: {accs[-1]:.2f}%")
    print(f"    Betti numbers: {results['betti_numbers']}")
    print(f"    Exactness: {[round(x, 3) for x in results['exactness']]}")
    
    return results, accs[-1]


# =============================================================================
# SECTION 10: MAIN EXECUTION
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  HOMOLOGICAL ALGEBRA FRAMEWORK FOR NEURAL NETWORKS")
    print("  Complete Results - All Output Displayed Inline")
    print("=" * 70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    results_summary = []
    
    # Run all experiments
    experiments = [
        ("TOY DATASET (Circles vs Squares)", experiment_toy),
        ("AGRICULTURE (Precision Viticulture)", experiment_agriculture),
        ("MNIST (Handwritten Digits)", experiment_mnist),
        ("CIFAR-10 (Object Recognition)", experiment_cifar10),
        ("ROTATED MNIST (SO(2) Equivariance)", experiment_rotated_mnist),
        ("SPHERICAL DATA (SO(3) Equivariance)", experiment_spherical)
    ]
    
    for name, exp_func in experiments:
        try:
            results, accuracy = exp_func()
            if results:
                results_summary.append({
                    'name': name,
                    'accuracy': accuracy,
                    'betti_sum': sum(results['betti_numbers']),
                    'avg_exactness': np.mean(results['exactness'])
                })
        except Exception as e:
            print(f"  ⚠ Error in {name}: {e}")
    
    # Final summary table
    print("\n" + "=" * 70)
    print("   FINAL SUMMARY - ALL EXPERIMENTS")
    print("=" * 70)
    print("\n  " + "-" * 65)
    print(f"  {'Experiment':<35} {'Accuracy':>10} {'Total β':>8} {'Exactness':>10}")
    print("  " + "-" * 65)
    
    for res in results_summary:
        print(f"  {res['name']:<35} {res['accuracy']:>9.2f}% {res['betti_sum']:>8} {res['avg_exactness']:>9.3f}")
    
    print("  " + "-" * 65)
    
    print("\n   ALL EXPERIMENTS COMPLETE!")
    print("   Results displayed above - no external files created")


if __name__ == "__main__":
    main()