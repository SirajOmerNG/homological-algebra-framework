"""
=============================================================================
  Homological Algebra Framework for Interpretable Deep Representations
  ── SINGLE-FILE VERSION FOR SPYDER ──────────────────────────────────────

  HOW TO RUN:
    1. Open this file in Spyder
    2. Press F5  (or Run → Run File)
    3. Watch the IPython console — everything runs automatically

  FIRST RUN ONLY:
    The script will download MNIST (~11 MB) and CIFAR-10 (~163 MB)
    automatically to a  ./data/  folder next to this file.

  OUTPUT:
    ./results/mnist/      — betti_numbers.png, exactness_analysis.png,
                            stability_analysis.png, training_curves.png
    ./results/cifar10/    — same set of plots
    ./results/comparison.png

  REQUIREMENTS  (install once in Anaconda Prompt):
    conda activate homological          # or your env name
    conda install pytorch torchvision -c pytorch -y
    conda install numpy scipy matplotlib -y

  Python 3.8+  |  tested with PyTorch 2.x
=============================================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
# ── Pick the right backend for Spyder ────────────────────────────────────────
# Spyder shows plots in its Plots pane automatically when using the default
# backend.  We only fall back to Agg (file-only) if no display is available.
try:
    import IPython
    shell = IPython.get_ipython()
    if shell is not None:
        # Running inside Spyder / Jupyter — use inline or Qt backend
        shell.run_line_magic('matplotlib', 'inline')
    else:
        raise RuntimeError("not in IPython")
except Exception:
    # Plain script execution (e.g. python script.py in terminal) — use Qt
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        matplotlib.use("Agg")   # last resort: save to files only
import matplotlib.pyplot as plt

# sklearn for toy + agriculture datasets (no download needed)
try:
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠  scikit-learn not found — Toy and Agriculture experiments will be skipped.")
    print("   Install: conda install scikit-learn -y")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    print("=" * 70)
    print("ERROR: PyTorch / TorchVision not found.")
    print("Open Anaconda Prompt and run:")
    print("  conda install pytorch torchvision -c pytorch -y")
    print("  conda install numpy scipy matplotlib -y")
    print("Then restart Spyder's kernel and press F5 again.")
    print("=" * 70)
    sys.exit(1)


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

class Config:
    """Central place to tweak every experiment parameter."""

    # Paths
    DATA_DIR    = r"C:\Users\Superuser\Desktop\homological\data"
    RESULTS_DIR = r"C:\Users\Superuser\Desktop\homological\results"
    
    # Training
    MNIST_EPOCHS   = 10
    CIFAR_EPOCHS   = 15
    BATCH_SIZE     = 128
    LEARNING_RATE  = 0.001

    # Analysis
    N_STABILITY_SAMPLES = 50   # reduce to 20 for faster runs
    MAX_CHAIN_SAMPLES   = 1000

    # Network architectures
    MNIST_DIMS  = [784,  128,  64,  32, 10]
    CIFAR_DIMS  = [3072, 512, 256, 128, 10]
    TOY_DIMS    = [2,    16,   8,        2]   # 2-D synthetic → 2 classes
    AGRI_DIMS   = [13,   32,  16,   8,   3]   # 13 chemical features → 3 cultivars
    TOY_EPOCHS  = 1000                         # fast: synthetic data
    AGRI_EPOCHS = 200                          # small tabular dataset


# =============================================================================
# SECTION 2 — DATASET LOADING
# =============================================================================

def load_mnist(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(
        Config.DATA_DIR, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(
        Config.DATA_DIR, train=False, download=True, transform=transform)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False))


def load_cifar10(batch_size: int = 128):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_test  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        Config.DATA_DIR, train=True,  download=True, transform=tf_train)
    test_ds  = torchvision.datasets.CIFAR10(
        Config.DATA_DIR, train=False, download=True, transform=tf_test)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False))


# ── Toy & Agriculture helpers (numpy/sklearn — no download) ──────────────────

def make_toy_dataset(n_samples: int = 1000,
                     noise: float = 0.05) -> Tuple[DataLoader, DataLoader]:
    """
    2-D synthetic dataset: circles vs squares (paper Section 5.1).
    Generates points inside a circle (class 0) and inside a square (class 1).
    """
    rng = np.random.default_rng(42)

    # Class 0: points inside unit circle
    angles  = rng.uniform(0, 2 * np.pi, n_samples // 2)
    radii   = np.sqrt(rng.uniform(0, 1,  n_samples // 2))
    circles = np.stack([radii * np.cos(angles),
                        radii * np.sin(angles)], axis=1)

    # Class 1: points inside [-0.9, 0.9]^2 square (excluding circle interior)
    squares = []
    while len(squares) < n_samples // 2:
        pts = rng.uniform(-0.9, 0.9, (n_samples, 2))
        outside = pts[np.sqrt(pts[:, 0]**2 + pts[:, 1]**2) > 1.0]
        squares.append(outside)
    squares = np.vstack(squares)[:n_samples // 2]

    X = np.vstack([circles, squares]).astype(np.float32)
    X += rng.normal(0, noise, X.shape).astype(np.float32)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int64)

    # Shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Train/test split 80/20
    split = int(0.8 * len(X))
    X_train, X_test = torch.tensor(X[:split]), torch.tensor(X[split:])
    y_train, y_test = torch.tensor(y[:split]), torch.tensor(y[split:])

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    test_ds  = torch.utils.data.TensorDataset(X_test,  y_test)
    return (DataLoader(train_ds, batch_size=64, shuffle=True),
            DataLoader(test_ds,  batch_size=64, shuffle=False))


def load_agriculture(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Precision viticulture / soil chemistry dataset (UCI Wine via sklearn).

    13 physicochemical soil/grape features measured from 3 cultivars grown
    in the same region of Italy.  Represents a real-world precision
    agriculture classification task:
        - alcohol, malic acid, ash, alkalinity, magnesium (soil/mineral)
        - phenols, flavanoids, proanthocyanins (plant chemistry)
        - colour intensity, hue, OD280 ratio, proline (harvest quality)
    Target: 3 grape cultivar classes → crop variety identification.

    Available offline via sklearn — no download needed.
    """
    if not SKLEARN_OK:
        raise ImportError("scikit-learn required: conda install scikit-learn -y")

    wine   = load_wine()
    scaler = StandardScaler()
    X      = scaler.fit_transform(wine.data).astype(np.float32)
    y      = wine.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    def to_loader(Xd, yd, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(Xd), torch.tensor(yd))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return to_loader(X_train, y_train, True), to_loader(X_test, y_test, False)


# =============================================================================
# SECTION 3 — CORE HOMOLOGICAL ALGEBRA
# =============================================================================

class ChainComplex:
    """
    A neural network modelled as a chain complex.

    C: 0 → C_0 --∂_1--> C_1 --∂_2--> … --∂_L--> C_L → 0

    ∂_l  is the weight matrix of layer l  (shape: d_{l+1} × d_l).
    Homology:   H_l = ker(∂_l) / im(∂_{l+1})
    Cohomology: H^l = ker(δ^l) / im(δ^{l-1}),   δ^l = (∂_l)^T
    Betti:      β_l = dim ker(∂_l) − rank(∂_{l+1})
    """

    def __init__(self,
                 representation_spaces: List[np.ndarray],
                 boundary_operators:    List[np.ndarray]):
        self.spaces     = representation_spaces
        self.boundaries = boundary_operators
        self.n_layers   = len(representation_spaces)

    # ── chain property ───────────────────────────────────────────────────────
    def verify_chain_property(self) -> Dict[int, float]:
        """
        Compute || ∂_{l+1} ∘ ∂_l ||_F for each consecutive pair.
        Near-zero → chain property holds (expected after training).
        """
        return {
            l: float(np.linalg.norm(self.boundaries[l + 1] @ self.boundaries[l], 'fro'))
            for l in range(len(self.boundaries) - 1)
        }

    # ── kernel & image via SVD ────────────────────────────────────────────────
    def compute_kernel(self,
                       A: np.ndarray,
                       tol: float = 1e-8) -> Tuple[np.ndarray, int]:
        """Right null-space of A.  Returns (basis_cols_in_R^ncols, dim)."""
        _, S, Vh = np.linalg.svd(A, full_matrices=True)
        rank  = int(np.sum(S > tol))
        ndim  = Vh.shape[0] - rank
        return Vh[rank:].T, ndim          # shape (n_cols, ndim)

    def compute_image(self,
                      A: np.ndarray,
                      tol: float = 1e-8) -> Tuple[np.ndarray, int]:
        """Column space of A.  Returns (basis_cols_in_R^nrows, dim)."""
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        rank = int(np.sum(S > tol))
        return U[:, :rank], rank          # shape (n_rows, rank)

    # ── homology ──────────────────────────────────────────────────────────────
    def compute_homology(self, l: int) -> Tuple[np.ndarray, int]:
        """
        β_l = dim ker(∂_l) − rank(∂_{l+1}).
        Kernel lives in R^{d_l}; we use the rank of the next boundary
        as the dimension of im(∂_{l+1}) inside the same space.
        """
        if l >= len(self.boundaries):
            return np.array([]), 0

        ker_basis, ker_dim = self.compute_kernel(self.boundaries[l])

        if l == len(self.boundaries) - 1:
            return ker_basis, ker_dim

        _, img_dim = self.compute_image(self.boundaries[l + 1])
        betti = max(0, ker_dim - img_dim)
        basis = ker_basis[:, :betti] if (betti > 0 and ker_basis.size > 0) else np.array([])
        return basis, betti

    # ── cohomology ────────────────────────────────────────────────────────────
    def compute_cohomology(self, l: int) -> Tuple[np.ndarray, int]:
        """
        β^l = dim ker(δ^l) − rank(δ^{l-1}),   δ^l = (∂_l)^T.
        """
        if l >= len(self.boundaries):
            return np.array([]), 0

        coker_basis, coker_dim = self.compute_kernel(self.boundaries[l].T)

        if l == 0:
            return coker_basis, coker_dim

        _, coimg_dim = self.compute_image(self.boundaries[l - 1].T)
        betti = max(0, coker_dim - coimg_dim)
        basis = coker_basis[:, :betti] if (betti > 0 and coker_basis.size > 0) else np.array([])
        return basis, betti


# =============================================================================
# SECTION 4 — EXACT SEQUENCE ANALYSIS
# =============================================================================

class ExactSequenceAnalyzer:
    """Measures how well information flows through the chain."""

    @staticmethod
    def compute_exactness(cc: ChainComplex, l: int) -> float:
        """
        Exactness ∈ [0, 1].  1.0 = ker(∂_l) == im(∂_{l+1})  (perfect flow).
        """
        if l >= len(cc.boundaries):
            return 1.0
        _, kd = cc.compute_kernel(cc.boundaries[l])
        if l == len(cc.boundaries) - 1:
            id_ = 0
        else:
            _, id_ = cc.compute_image(cc.boundaries[l + 1])
        return 1.0 - abs(kd - id_) / max(kd, id_, 1)

    @staticmethod
    def compute_information_loss(cc: ChainComplex, l: int) -> int:
        """L_l = ker_dim − img_dim.  >0 → loss; <0 → expansion."""
        if l >= len(cc.boundaries):
            return 0
        _, kd = cc.compute_kernel(cc.boundaries[l])
        if l == len(cc.boundaries) - 1:
            id_ = 0
        else:
            _, id_ = cc.compute_image(cc.boundaries[l + 1])
        return kd - id_

    @staticmethod
    def analyze_all_layers(cc: ChainComplex) -> Dict[str, List]:
        results: Dict[str, List] = {
            'exactness': [], 'information_loss': [],
            'betti_numbers_homology': [], 'betti_numbers_cohomology': [],
        }
        for l in range(len(cc.boundaries)):
            results['exactness'].append(ExactSequenceAnalyzer.compute_exactness(cc, l))
            results['information_loss'].append(ExactSequenceAnalyzer.compute_information_loss(cc, l))
            _, bh = cc.compute_homology(l)
            _, bc = cc.compute_cohomology(l)
            results['betti_numbers_homology'].append(bh)
            results['betti_numbers_cohomology'].append(bc)
        return results


# =============================================================================
# SECTION 5 — BAYESIAN STABILITY ANALYSIS
# =============================================================================

class BayesianStabilityAnalyzer:
    """
    Perturbs weight matrices with Gaussian noise and tracks Betti number variance.
    S_l = 1 / (Var(β_l) + ε).  Higher → more robust topology.
    """

    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std

    def perturb_weights(self, W: np.ndarray) -> np.ndarray:
        return W + np.random.normal(0, self.noise_std, W.shape)

    def compute_stability_metrics(self,
                                  cc: ChainComplex,
                                  n_samples: int = 50) -> Dict[str, dict]:
        betti_samples: Dict[str, List[int]] = defaultdict(list)

        for _ in range(n_samples):
            perturbed = [self.perturb_weights(b) for b in cc.boundaries]
            sample_cc = ChainComplex(cc.spaces, perturbed)
            for l in range(len(perturbed)):
                _, bh = sample_cc.compute_homology(l)
                _, bc = sample_cc.compute_cohomology(l)
                betti_samples[f'homology_{l}'].append(bh)
                betti_samples[f'cohomology_{l}'].append(bc)

        results = {}
        for key, vals in betti_samples.items():
            arr = np.array(vals, dtype=float)
            var = float(np.var(arr))
            results[key] = {
                'mean':      float(np.mean(arr)),
                'variance':  var,
                'stability': 1.0 / (var + 1e-8),
                'samples':   arr,
            }
        return results


# =============================================================================
# SECTION 6 — NEURAL NETWORK
# =============================================================================

class HomologicalNN(nn.Module):
    """Fully-connected network with built-in chain-complex extraction."""

    def __init__(self, layer_dims: List[int], activation: str = 'relu'):
        super().__init__()
        self.layer_dims = layer_dims
        self.n_layers   = len(layer_dims) - 1
        self.layers     = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(self.n_layers)
        ])
        _act = {'relu': F.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}
        if activation not in _act:
            raise ValueError(f"Unknown activation '{activation}'")
        self.activation      = _act[activation]
        self.activation_name = activation
        self.representations: List[np.ndarray] = []

    def forward(self, x: torch.Tensor,
                store: bool = False) -> torch.Tensor:
        if store:
            self.representations = [x.detach().cpu().numpy()]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self.n_layers - 1:
                h = self.activation(h)
            if store:
                self.representations.append(h.detach().cpu().numpy())
        return h

    def extract_chain_complex(self,
                              loader: DataLoader,
                              max_samples: int = 1000) -> ChainComplex:
        self.eval()
        all_reps: List[List[np.ndarray]] = [[] for _ in range(self.n_layers + 1)]
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

    def analyze(self,
                loader: DataLoader,
                n_stability_samples: int = 50) -> Dict:
        cc            = self.extract_chain_complex(loader, Config.MAX_CHAIN_SAMPLES)
        exact         = ExactSequenceAnalyzer.analyze_all_layers(cc)
        stability     = BayesianStabilityAnalyzer().compute_stability_metrics(
                            cc, n_samples=n_stability_samples)
        chain_viol    = cc.verify_chain_property()
        return {
            'exact_sequence_analysis': exact,
            'stability_metrics':       stability,
            'chain_property_violations': chain_viol,
            'chain_complex':           cc,
        }


# =============================================================================
# SECTION 7 — TRAINING
# =============================================================================

def train_network(model: HomologicalNN,
                  train_loader: DataLoader,
                  test_loader:  DataLoader,
                  epochs: int   = 10,
                  lr:     float = 0.001,
                  device: str   = 'cpu',
                  label:  str   = '') -> Tuple[List[float], List[float]]:

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining {label} on {device} for {epochs} epochs …")
    print("-" * 70)

    train_losses, test_accs = [], []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for bx, by in train_loader:
            bx = bx.view(bx.size(0), -1).to(device)
            by = by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            running += loss.item()

        avg_loss = running / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.view(bx.size(0), -1).to(device)
                by = by.to(device)
                preds = model(bx).argmax(1)
                total   += by.size(0)
                correct += (preds == by).sum().item()

        acc = 100 * correct / total
        test_accs.append(acc)
        print(f"  Epoch {epoch+1:2d}/{epochs}:  loss = {avg_loss:.4f},  "
              f"test accuracy = {acc:.2f}%")

    model = model.cpu()
    print(f"\n  ✓ Final accuracy: {test_accs[-1]:.2f}%")
    return train_losses, test_accs


# =============================================================================
# SECTION 8 — VISUALISATION
# =============================================================================

def _save(fig: plt.Figure, path: str) -> None:
    """Save figure then close it to free memory."""
    fig.savefig(path, dpi=300, bbox_inches='tight')
    # figures stay open so Spyder/inline can display them


def plot_training_curves(losses: List[float],
                         accs:   List[float],
                         title:  str,
                         save_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(losses, linewidth=2)
    ax1.set(xlabel='Epoch', ylabel='Training Loss', title=f'{title} Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(accs, linewidth=2, color='green')
    ax2.set(xlabel='Epoch', ylabel='Test Accuracy (%)', title=f'{title} Test Accuracy')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, 'training_curves.png'))


def plot_betti_numbers(results: Dict, save_dir: str) -> None:
    ea = results['exact_sequence_analysis']
    layers = list(range(len(ea['betti_numbers_homology'])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(layers, ea['betti_numbers_homology'],
             marker='o', linewidth=2, markersize=8, label='Homology')
    ax1.set(xlabel='Layer', ylabel='Betti Number (β_l)',
            title='Homology Betti Numbers')
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.plot(layers, ea['betti_numbers_cohomology'],
             marker='s', linewidth=2, markersize=8,
             color='orange', label='Cohomology')
    ax2.set(xlabel='Layer', ylabel='Betti Number (β_l)',
            title='Cohomology Betti Numbers')
    ax2.grid(True, alpha=0.3); ax2.legend()

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, 'betti_numbers.png'))


def plot_exactness(results: Dict, save_dir: str) -> None:
    ea = results['exact_sequence_analysis']
    layers = list(range(len(ea['exactness'])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(layers, ea['exactness'], marker='o', linewidth=2,
             markersize=8, color='green')
    ax1.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Perfect')
    ax1.set(xlabel='Layer', ylabel='Exactness',
            title='Exact Sequence Analysis', ylim=[0, 1.1])
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.bar(layers, ea['information_loss'], color='coral', alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set(xlabel='Layer', ylabel='Information Loss (L_l)',
            title='Information Flow Analysis')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, 'exactness_analysis.png'))


def plot_stability(results: Dict, save_dir: str) -> None:
    sm = results['stability_metrics']
    layers, stabs, vars_ = [], [], []
    for key in sorted(sm):
        if 'homology' in key:
            layers.append(int(key.split('_')[1]))
            stabs.append(sm[key]['stability'])
            vars_.append(sm[key]['variance'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(layers, stabs, marker='o', linewidth=2, markersize=8, color='purple')
    ax1.set(xlabel='Layer', ylabel='Stability (S_l)',
            title='Cohomological Stability')
    ax1.set_yscale('log'); ax1.grid(True, alpha=0.3)

    ax2.bar(layers, vars_, color='steelblue', alpha=0.7)
    ax2.set(xlabel='Layer', ylabel='Variance of Betti Numbers',
            title='Topological Feature Variance')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, 'stability_analysis.png'))


def generate_all_plots(results: Dict, save_dir: str,
                       train_losses: List[float],
                       test_accs:    List[float],
                       title: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    plot_training_curves(train_losses, test_accs, title, save_dir)
    plot_betti_numbers(results, save_dir)
    plot_exactness(results, save_dir)
    plot_stability(results, save_dir)
    print(f"  ✓ Plots saved → {save_dir}")


def plot_comparison(mr: Dict, cr: Dict) -> None:
    """Side-by-side MNIST vs CIFAR-10 comparison."""
    mea = mr['exact_sequence_analysis']
    cea = cr['exact_sequence_analysis']
    msm = mr['stability_metrics']
    csm = cr['stability_metrics']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Betti numbers
    axes[0, 0].plot(mea['betti_numbers_homology'], marker='o', label='MNIST')
    axes[0, 0].plot(cea['betti_numbers_homology'], marker='s', label='CIFAR-10')
    axes[0, 0].set(title='Betti Numbers (Homology)',
                   xlabel='Layer', ylabel='β_l')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # Exactness
    axes[0, 1].plot(mea['exactness'], marker='o', label='MNIST')
    axes[0, 1].plot(cea['exactness'], marker='s', label='CIFAR-10')
    axes[0, 1].set(title='Exactness', xlabel='Layer', ylabel='Exactness')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Information loss
    lm = list(range(len(mea['information_loss'])))
    lc = list(range(len(cea['information_loss'])))
    axes[1, 0].bar(np.array(lm) - 0.2, mea['information_loss'],
                   width=0.4, label='MNIST', alpha=0.7)
    axes[1, 0].bar(np.array(lc) + 0.2, cea['information_loss'],
                   width=0.4, label='CIFAR-10', alpha=0.7)
    axes[1, 0].axhline(0, color='black', linewidth=0.5)
    axes[1, 0].set(title='Information Loss', xlabel='Layer', ylabel='L_l')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Stability
    mv = [msm[k]['stability'] for k in sorted(msm) if 'homology' in k]
    cv = [csm[k]['stability'] for k in sorted(csm) if 'homology' in k]
    axes[1, 1].plot(mv, marker='o', label='MNIST')
    axes[1, 1].plot(cv, marker='s', label='CIFAR-10')
    axes[1, 1].set(title='Stability (log)', xlabel='Layer', ylabel='S_l')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(Config.RESULTS_DIR, 'comparison.png')
    _save(fig, path)
    print(f"  ✓ Comparison plot saved → {path}")


# =============================================================================
# SECTION 9 — SUMMARY PRINTER
# =============================================================================

def print_summary(results: Dict, label: str = '') -> None:
    ea = results['exact_sequence_analysis']
    sm = results['stability_metrics']
    cv = results['chain_property_violations']

    print(f"\n{'=' * 70}")
    print(f"  HOMOLOGICAL ANALYSIS SUMMARY  —  {label}")
    print(f"{'=' * 70}")

    print("\n  [BETTI NUMBERS]")
    for i, (bh, bc) in enumerate(zip(ea['betti_numbers_homology'],
                                     ea['betti_numbers_cohomology'])):
        print(f"    Layer {i}: β_h = {bh:3d},  β_c = {bc:3d}")

    print("\n  [EXACTNESS & INFORMATION LOSS]")
    for i, (ex, loss) in enumerate(zip(ea['exactness'], ea['information_loss'])):
        tag = "✓ exact" if ex > 0.9 else "✗ non-exact"
        print(f"    Layer {i}: exactness = {ex:.3f}  ({tag}),  loss = {loss:+d}")
    bns = [i for i, e in enumerate(ea['exactness']) if e < 0.7]
    if bns:
        print(f"\n  ⚠  Bottleneck layers: {bns}")

    print("\n  [CHAIN PROPERTY  ‖∂_{l+1}∘∂_l‖_F]")
    for l, v in sorted(cv.items()):
        print(f"    l={l}: {v:.4e}")

    print("\n  [STABILITY]")
    for key in sorted(sm):
        if 'homology' in key:
            lyr = key.split('_')[1]
            m   = sm[key]
            print(f"    Layer {lyr}: S = {m['stability']:.2f},  "
                  f"Var = {m['variance']:.4f},  mean β = {m['mean']:.2f}")

    print(f"\n{'=' * 70}")


# =============================================================================
# SECTION 10 — EXPERIMENTS
# =============================================================================

def run_mnist_experiment(device: str) -> Tuple[HomologicalNN, Dict,
                                               List[float], List[float]]:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: MNIST")
    print("=" * 70)
    print(f"  Architecture: {Config.MNIST_DIMS}")

    train_loader, test_loader = load_mnist(Config.BATCH_SIZE)

    model = HomologicalNN(Config.MNIST_DIMS, activation='relu')
    losses, accs = train_network(
        model, train_loader, test_loader,
        epochs=Config.MNIST_EPOCHS, lr=Config.LEARNING_RATE,
        device=device, label='MNIST')

    print("\n  Performing homological analysis (may take a few minutes)…")
    results = model.analyze(test_loader,
                            n_stability_samples=Config.N_STABILITY_SAMPLES)
    print_summary(results, 'MNIST')

    save_dir = os.path.join(Config.RESULTS_DIR, 'mnist')
    generate_all_plots(results, save_dir, losses, accs, 'MNIST')
    return model, results, losses, accs


def run_cifar10_experiment(device: str) -> Tuple[HomologicalNN, Dict,
                                                 List[float], List[float]]:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: CIFAR-10")
    print("=" * 70)
    print(f"  Architecture: {Config.CIFAR_DIMS}")

    train_loader, test_loader = load_cifar10(Config.BATCH_SIZE)

    model = HomologicalNN(Config.CIFAR_DIMS, activation='relu')
    losses, accs = train_network(
        model, train_loader, test_loader,
        epochs=Config.CIFAR_EPOCHS, lr=Config.LEARNING_RATE,
        device=device, label='CIFAR-10')

    print("\n  Performing homological analysis (may take a few minutes)…")
    results = model.analyze(test_loader,
                            n_stability_samples=Config.N_STABILITY_SAMPLES)
    print_summary(results, 'CIFAR-10')

    save_dir = os.path.join(Config.RESULTS_DIR, 'cifar10')
    generate_all_plots(results, save_dir, losses, accs, 'CIFAR-10')
    return model, results, losses, accs


# =============================================================================
# SECTION 11 — TOY EXPERIMENT  (paper Section 5.1 — circles vs squares)
# =============================================================================

def run_toy_experiment(device: str):
    """
    Reproduces paper Table 1 (Section 5.1).
    Tiny 3-layer network on 2-D synthetic shapes — runs in < 1 minute.
    Expected results:
        β = [1, 2, 1]   information loss ≈ [0%, 20%, 5%]
        stability ≈ [0.89, 0.72, 0.94]
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: TOY NETWORK  (circles vs squares, paper §5.1)")
    print("=" * 70)
    print(f"  Architecture: {Config.TOY_DIMS}")
    print(  "  Data: 1000 synthetic 2-D samples (500 circles, 500 squares)")

    train_loader, test_loader = make_toy_dataset(n_samples=1000)

    model  = HomologicalNN(Config.TOY_DIMS, activation='relu')
    losses, accs = train_network(
        model, train_loader, test_loader,
        epochs=Config.TOY_EPOCHS, lr=Config.LEARNING_RATE,
        device=device, label='Toy')

    print("\n  Performing homological analysis…")
    results = model.analyze(test_loader,
                            n_stability_samples=Config.N_STABILITY_SAMPLES)
    print_summary(results, 'Toy Network (circles vs squares)')

    # ── extra: reproduce Table 1 numbers ─────────────────────────────────────
    ea = results['exact_sequence_analysis']
    sm = results['stability_metrics']
    print("\n  TABLE 1 REPRODUCTION  (paper §5.1):")
    print(f"  {'Metric':<25} {'Layer 1':>10} {'Layer 2':>10} {'Layer 3':>10}")
    print("  " + "-" * 55)
    print(f"  {'Betti number (β_l)':<25}",
          *[f"{b:>10}" for b in ea['betti_numbers_homology']])
    info_pct = [f"{abs(l)/max(abs(l),1)*100:.0f}%"
                if l != 0 else "0%"
                for l in ea['information_loss']]
    print(f"  {'Information loss':<25}",
          *[f"{p:>10}" for p in info_pct])
    stab_vals = [sm[f'homology_{i}']['stability'] for i in range(len(ea['exactness']))]
    # normalise to 0-1 range for interpretability
    max_s = max(stab_vals) if max(stab_vals) > 0 else 1
    stab_norm = [f"{min(s/max_s, 1.0):.2f}" for s in stab_vals]
    print(f"  {'Stability (S_l, normalised)':<25}",
          *[f"{s:>10}" for s in stab_norm])

    save_dir = os.path.join(Config.RESULTS_DIR, 'toy')
    generate_all_plots(results, save_dir, losses, accs, 'Toy Network')

    # ── decision boundary visualisation ──────────────────────────────────────
    _plot_toy_boundary(model, save_dir)

    return model, results, losses, accs


def _plot_toy_boundary(model: HomologicalNN, save_dir: str) -> None:
    """Visualise the learned decision boundary on 2-D toy data."""
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 300),
                         np.linspace(-1.5, 1.5, 300))
    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(grid)
        zz = logits.argmax(1).numpy().reshape(xx.shape)

    # re-generate data for scatter
    rng    = np.random.default_rng(42)
    angles = rng.uniform(0, 2*np.pi, 500)
    radii  = np.sqrt(rng.uniform(0, 1, 500))
    circ   = np.stack([radii*np.cos(angles), radii*np.sin(angles)], 1)
    sq_all = rng.uniform(-0.9, 0.9, (5000, 2))
    sq     = sq_all[np.sqrt(sq_all[:,0]**2 + sq_all[:,1]**2) > 1.0][:500]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(xx, yy, zz, alpha=0.25, cmap='RdBu')
    ax.scatter(circ[:,0], circ[:,1], s=8, c='royalblue',  label='Circle (class 0)')
    ax.scatter(sq[:,0],   sq[:,1],   s=8, c='tomato',     label='Square (class 1)')
    ax.set_title('Toy Network — Learned Decision Boundary', fontweight='bold')
    ax.legend(loc='upper right'); ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, 'decision_boundary.png'))


# =============================================================================
# SECTION 12 — AGRICULTURE EXPERIMENT  (precision viticulture)
# =============================================================================

def run_agriculture_experiment(device: str):
    """
    Precision Agriculture / Viticulture Experiment.

    Dataset : UCI Wine (sklearn, no download needed)
    Context : 13 physicochemical soil & grape measurements from 3 cultivars
              grown in the same Italian region — a real precision-agriculture
              crop-variety identification task.
    Network : [13 → 32 → 16 → 8 → 3]

    Homological interpretation:
        H^1  — chemical features invariant to seasonal variation (pH, minerals)
        H^2  — composite soil-quality signatures
        Stability — which chemical markers reliably distinguish cultivars

    Why this matters for agriculture:
        Non-exact sequences reveal which transformation steps lose diagnostic
        soil information; stability metrics show which markers are robust
        across different harvests / measurement conditions.
    """
    if not SKLEARN_OK:
        print("\n  ⚠  Skipping Agriculture experiment — scikit-learn not installed.")
        print("     Run: conda install scikit-learn -y  then restart Spyder kernel.")
        return None, None, None, None

    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: PRECISION AGRICULTURE  (viticulture / soil chemistry)")
    print("=" * 70)
    print(f"  Architecture: {Config.AGRI_DIMS}")
    print(  "  Dataset: UCI Wine — 178 samples, 13 physicochemical features")
    print(  "  Task: Identify grape cultivar from soil & harvest chemistry")
    print()
    print("  Features (agriculture context):")
    feature_labels = [
        "Alcohol content",          "Malic acid  (soil acidity)",
        "Ash  (mineral content)",   "Ash alkalinity  (soil pH proxy)",
        "Magnesium  (soil mineral)","Total phenols  (antioxidant)",
        "Flavanoids  (plant chem)", "Nonflavanoid phenols",
        "Proanthocyanins",          "Colour intensity",
        "Hue  (harvest quality)",   "OD280/OD315  (protein content)",
        "Proline  (amino acid)",
    ]
    for i, lbl in enumerate(feature_labels):
        print(f"    {i+1:2d}. {lbl}")

    train_loader, test_loader = load_agriculture()

    model  = HomologicalNN(Config.AGRI_DIMS, activation='relu')
    losses, accs = train_network(
        model, train_loader, test_loader,
        epochs=Config.AGRI_EPOCHS, lr=Config.LEARNING_RATE,
        device=device, label='Agriculture')

    print("\n  Performing homological analysis…")
    results = model.analyze(test_loader,
                            n_stability_samples=Config.N_STABILITY_SAMPLES)
    print_summary(results, 'Agriculture (Precision Viticulture)')

    # ── agriculture-specific interpretation ──────────────────────────────────
    ea = results['exact_sequence_analysis']
    sm = results['stability_metrics']
    print("\n  AGRICULTURE INTERPRETATION:")
    for l, (ex, loss) in enumerate(zip(ea['exactness'], ea['information_loss'])):
        stab = sm[f'homology_{l}']['stability']
        if l == 0:
            layer_name = "Soil mineral encoding   (input → 32-D)"
        elif l == 1:
            layer_name = "Chemical interaction    (32-D → 16-D)"
        elif l == 2:
            layer_name = "Quality signature       (16-D → 8-D)"
        else:
            layer_name = "Cultivar classification (8-D → output)"
        tag = "✓ lossless" if ex > 0.9 else ("⚠ bottleneck" if ex < 0.7 else "~ partial loss")
        print(f"    Layer {l} [{layer_name}]: {tag}  |  "
              f"exactness={ex:.3f}  loss={loss:+d}  S={stab:.1f}")

    save_dir = os.path.join(Config.RESULTS_DIR, 'agriculture')
    generate_all_plots(results, save_dir, losses, accs, 'Agriculture')

    # ── feature importance from cohomology ───────────────────────────────────
    _plot_agri_feature_importance(results, feature_labels, save_dir)

    return model, results, losses, accs


def _plot_agri_feature_importance(results: Dict,
                                   feature_labels: List[str],
                                   save_dir: str) -> None:
    """
    Visualise which input features matter most based on the first-layer
    weight matrix (proxy for cohomological feature importance).
    """
    cc = results['chain_complex']
    W1 = cc.boundaries[0]   # shape (32, 13)
    importance = np.linalg.norm(W1, axis=0)   # L2 norm of each input column
    importance /= importance.sum()             # normalise to sum = 1

    short_labels = [
        "Alcohol","Malic acid","Ash","Ash alkalinity","Magnesium",
        "Phenols","Flavanoids","NF Phenols","Proanthocyan.","Colour",
        "Hue","OD280","Proline",
    ]
    colours = ['steelblue' if imp > np.median(importance) else 'lightsteelblue'
               for imp in importance]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(13), importance * 100, color=colours)
    ax.set_xticks(range(13))
    ax.set_xticklabels(short_labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Relative Importance (%)')
    ax.set_title('Agriculture: Input Feature Importance\n'
                 '(L2 norm of first-layer weight columns)',
                 fontweight='bold')
    ax.axhline(100/13, color='red', linestyle='--', alpha=0.6,
               label=f'Uniform baseline ({100/13:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, 'feature_importance.png'))


# =============================================================================
# EXECUTION  — just press F5 in Spyder
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  Homological Algebra Framework for Neural Network Interpretability")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    # ── device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device.upper()}")

    # ── output directories ────────────────────────────────────────────────────
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.DATA_DIR,    exist_ok=True)

    # ── Experiment order ─────────────────────────────────────────────────────
    # EXP 1: Toy network       ~  1 min  | no download, sklearn not needed
    # EXP 2: Agriculture       ~  1 min  | no download, needs scikit-learn
    # EXP 3: MNIST             ~  5-10 min | downloads ~11 MB on first run
    # EXP 4: CIFAR-10          ~ 10-15 min | downloads ~163 MB on first run
    # ─────────────────────────────────────────────────────────────────────────

    try:
        # ── EXP 1: Toy network (paper §5.1 — fastest, no download) ───────────
        toy_model,   toy_results,   toy_losses,   toy_accs   = run_toy_experiment(device)

        # ── EXP 2: Agriculture / precision viticulture ────────────────────────
        agri_model,  agri_results,  agri_losses,  agri_accs  = run_agriculture_experiment(device)

        # ── EXP 3: MNIST ─────────────────────────────────────────────────────
        mnist_model, mnist_results, mnist_losses, mnist_accs = run_mnist_experiment(device)

        # ── EXP 4: CIFAR-10 ──────────────────────────────────────────────────
        cifar_model, cifar_results, cifar_losses, cifar_accs = run_cifar10_experiment(device)

        # ── comparative plot (MNIST vs CIFAR-10) ──────────────────────────────
        print("\n  Generating MNIST vs CIFAR-10 comparative analysis…")
        plot_comparison(mnist_results, cifar_results)

        # ── all-experiments summary ───────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  ✅  ALL EXPERIMENTS COMPLETE")
        print("=" * 70)
        print(f"\n  Results saved to: {os.path.abspath(Config.RESULTS_DIR)}/")
        print("    toy/         — betti_numbers.png, exactness_analysis.png,")
        print("                   stability_analysis.png, training_curves.png,")
        print("                   decision_boundary.png")
        print("    agriculture/ — same plots + feature_importance.png")
        print("    mnist/       — same 4 plots")
        print("    cifar10/     — same 4 plots")
        print("    comparison.png  (MNIST vs CIFAR-10)")

        print("\n  KEY FINDINGS ACROSS ALL EXPERIMENTS:")
        print(f"  {'Experiment':<20} {'Total β':>9} {'Final Acc':>10} {'Avg Exact':>10}")
        print("  " + "-" * 53)
        for label, res, acc in [
            ("Toy network",    toy_results,   toy_accs),
            ("Agriculture",    agri_results,  agri_accs  if agri_accs  else [0]),
            ("MNIST",          mnist_results, mnist_accs),
            ("CIFAR-10",       cifar_results, cifar_accs),
        ]:
            if res is None:
                continue
            tb  = sum(res['exact_sequence_analysis']['betti_numbers_homology'])
            ae  = np.mean(res['exact_sequence_analysis']['exactness'])
            print(f"  {label:<20} {tb:>9}  {acc[-1]:>9.1f}%  {ae:>9.3f}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nCommon fixes:")
        print("  • No internet for MNIST/CIFAR-10?  Toy + Agriculture still run offline.")
        print("  • Out of memory?  Lower Config.BATCH_SIZE or Config.N_STABILITY_SAMPLES.")
        print("  • Missing torch?    conda install pytorch torchvision -c pytorch -y")
        print("  • Missing sklearn?  conda install scikit-learn -y")

    # ── display all figures in Spyder Plots pane ──────────────────────────────
    plt.show()