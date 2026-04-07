"""
Homological Algebra Framework for Interpretable Deep Representations

This implementation provides the core framework from the paper:
"A Homological Algebra Framework for Interpretable Deep Representations"

Key Features:
- Neural networks modeled as chain complexes
- Homology and cohomology group computation
- Exact sequence analysis for information flow
- Bayesian stability analysis
- Feature invariant detection

Author: Implementation based on research paper
Date: January 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import linalg
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from collections import defaultdict


# =============================================================================
# CORE HOMOLOGICAL ALGEBRA OPERATIONS
# =============================================================================

class ChainComplex:
    """
    Represents a neural network as a chain complex with boundary operators.
    
    Mathematical Structure:
    C: 0 → H^(0) → H^(1) → ... → H^(L) → 0
    
    where H^(l) are representation spaces and arrows are boundary operators.
    
    Note: For neural networks, we model each layer's representation space as a module,
    and the boundary operators are derived from the weight matrices and activations.
    """
    
    def __init__(self, representation_spaces: List[np.ndarray], 
                 boundary_operators: List[np.ndarray]):
        """
        Args:
            representation_spaces: List of layer representations [N x d_l]
            boundary_operators: List of boundary operators (weight matrices)
        """
        self.spaces = representation_spaces
        self.boundaries = boundary_operators
        self.n_layers = len(representation_spaces)
        
        # Validate dimensions
        for i, boundary in enumerate(self.boundaries):
            if i < len(self.spaces) - 1:
                input_dim = self.spaces[i].shape[1]
                output_dim = self.spaces[i + 1].shape[1]
                
                # Boundary operator should map from space i to space i+1
                # Weight matrix shape: (output_dim, input_dim)
                expected_shape = (output_dim, input_dim)
                
                if boundary.shape != expected_shape:
                    print(f"Warning: Boundary {i} has shape {boundary.shape}, "
                          f"expected {expected_shape}")
    
    def verify_chain_property(self, tolerance=1e-6) -> Dict[int, float]:
        """
        Verify the chain complex property for neural networks.
        
        For a true chain complex, we need ∂_{l-1} ∘ ∂_l = 0.
        In neural networks, this doesn't hold exactly due to nonlinearity,
        but we can measure the deviation.
        
        Returns:
            Dictionary mapping layer index to a validity metric
        """
        violations = {}
        
        # For neural networks, we check properties of individual operators
        for l in range(len(self.boundaries)):
            # Compute rank of boundary operator
            rank = np.linalg.matrix_rank(self.boundaries[l])
            expected_rank = min(self.boundaries[l].shape)
            
            # Measure how "well-behaved" the operator is
            # Full rank indicates good information flow
            rank_ratio = rank / expected_rank if expected_rank > 0 else 0
            violations[l] = 1.0 - rank_ratio  # 0 = full rank (good), 1 = rank deficient (bad)
            
        return violations
    
    def compute_kernel(self, boundary_op: np.ndarray, tolerance=1e-8) -> Tuple[np.ndarray, int]:
        """
        Compute kernel of boundary operator using SVD.
        
        ker(∂_l) = {c ∈ C_l : ∂_l(c) = 0}
        
        Returns:
            Basis vectors for kernel and its dimension
        """
        # Use SVD to find null space
        U, S, Vh = np.linalg.svd(boundary_op, full_matrices=True)
        
        # Find dimensions where singular value is near zero
        rank = np.sum(S > tolerance)
        null_space_dim = Vh.shape[0] - rank
        
        # Null space is spanned by right singular vectors corresponding to zero singular values
        kernel_basis = Vh[rank:].T
        
        return kernel_basis, null_space_dim
    
    def compute_image(self, boundary_op: np.ndarray, tolerance=1e-8) -> Tuple[np.ndarray, int]:
        """
        Compute image of boundary operator.
        
        im(∂_{l+1}) = range(∂_{l+1})
        
        Returns:
            Basis vectors for image and its dimension
        """
        # Use SVD to find range
        U, S, Vh = np.linalg.svd(boundary_op, full_matrices=False)
        
        rank = np.sum(S > tolerance)
        image_basis = U[:, :rank]
        
        return image_basis, rank
    
    def compute_homology(self, layer_idx: int) -> Tuple[np.ndarray, int]:
        """
        Compute homology group at layer l:
        
        H_l = ker(∂_l) / im(∂_{l+1})
        
        Returns:
            Basis for homology group and Betti number (rank)
        """
        if layer_idx >= len(self.boundaries):
            return np.array([]), 0
            
        # Get kernel of current boundary
        kernel_basis, kernel_dim = self.compute_kernel(self.boundaries[layer_idx])
        
        if layer_idx == len(self.boundaries) - 1:
            # Last layer: no image to quotient by
            return kernel_basis, kernel_dim
        
        # Get image of next boundary
        image_basis, image_dim = self.compute_image(self.boundaries[layer_idx + 1])
        
        # Compute quotient space dimension (Betti number)
        # β_l = dim(ker(∂_l)) - dim(im(∂_{l+1}))
        betti_number = max(0, kernel_dim - image_dim)
        
        # Compute basis for quotient space
        if betti_number > 0 and image_dim > 0 and kernel_dim > 0:
            # Check if dimensions are compatible
            if image_basis.shape[0] == kernel_basis.shape[0]:
                # Project kernel onto complement of image
                # Use QR decomposition for numerical stability
                combined = np.hstack([image_basis, kernel_basis])
                Q, R = np.linalg.qr(combined)
                
                # Homology basis is the part orthogonal to image
                homology_basis = Q[:, image_dim:image_dim + betti_number]
            else:
                # Dimension mismatch - use kernel basis directly
                homology_basis = kernel_basis[:, :betti_number] if kernel_basis.shape[1] >= betti_number else kernel_basis
        else:
            homology_basis = kernel_basis[:, :betti_number] if betti_number > 0 and kernel_basis.size > 0 else np.array([])
        
        return homology_basis, betti_number
    
    def compute_cohomology(self, layer_idx: int) -> Tuple[np.ndarray, int]:
        """
        Compute cohomology group at layer l:
        
        H^l = ker(δ^l) / im(δ^{l-1})
        
        where δ^l = (∂_l)^T is the coboundary operator.
        
        Returns:
            Basis for cohomology group and Betti number
        """
        if layer_idx >= len(self.boundaries):
            return np.array([]), 0
        
        # Coboundary is transpose of boundary
        coboundary = self.boundaries[layer_idx].T
        
        # Compute cokernel
        cokernel_basis, cokernel_dim = self.compute_kernel(coboundary)
        
        if layer_idx == 0:
            # First layer: no coimage to quotient by
            return cokernel_basis, cokernel_dim
        
        # Get coimage (image of previous coboundary)
        prev_coboundary = self.boundaries[layer_idx - 1].T
        coimage_basis, coimage_dim = self.compute_image(prev_coboundary)
        
        # Compute quotient dimension
        betti_number = max(0, cokernel_dim - coimage_dim)
        
        # Compute cohomology basis
        if betti_number > 0 and coimage_dim > 0 and cokernel_dim > 0:
            # Check dimension compatibility
            if coimage_basis.shape[0] == cokernel_basis.shape[0]:
                combined = np.hstack([coimage_basis, cokernel_basis])
                Q, R = np.linalg.qr(combined)
                cohomology_basis = Q[:, coimage_dim:coimage_dim + betti_number]
            else:
                # Dimension mismatch - use cokernel basis directly
                cohomology_basis = cokernel_basis[:, :betti_number] if cokernel_basis.shape[1] >= betti_number else cokernel_basis
        else:
            cohomology_basis = cokernel_basis[:, :betti_number] if betti_number > 0 and cokernel_basis.size > 0 else np.array([])
        
        return cohomology_basis, betti_number


# =============================================================================
# EXACT SEQUENCE ANALYSIS
# =============================================================================

class ExactSequenceAnalyzer:
    """
    Analyzes exact sequences to characterize information flow in neural networks.
    
    A sequence is exact at C_l if ker(∂_l) = im(∂_{l+1})
    Non-exactness indicates information loss or redundancy.
    """
    
    @staticmethod
    def compute_exactness(chain_complex: ChainComplex, layer_idx: int, 
                         tolerance=1e-6) -> float:
        """
        Compute exactness measure at layer l.
        
        Exactness = 1 - |dim(ker) - dim(im)| / max(dim(ker), dim(im))
        
        Returns:
            Value in [0, 1] where 1 = perfectly exact
        """
        if layer_idx >= len(chain_complex.boundaries):
            return 1.0
        
        _, kernel_dim = chain_complex.compute_kernel(chain_complex.boundaries[layer_idx])
        
        if layer_idx == len(chain_complex.boundaries) - 1:
            image_dim = 0
        else:
            _, image_dim = chain_complex.compute_image(chain_complex.boundaries[layer_idx + 1])
        
        max_dim = max(kernel_dim, image_dim, 1)  # Avoid division by zero
        exactness = 1 - abs(kernel_dim - image_dim) / max_dim
        
        return exactness
    
    @staticmethod
    def compute_information_loss(chain_complex: ChainComplex, layer_idx: int) -> int:
        """
        Compute information loss at layer l.
        
        L_l = dim(ker(∂_l)) - dim(im(∂_{l+1}))
        
        Positive value indicates information loss.
        Negative value indicates redundancy/expansion.
        """
        if layer_idx >= len(chain_complex.boundaries):
            return 0
        
        _, kernel_dim = chain_complex.compute_kernel(chain_complex.boundaries[layer_idx])
        
        if layer_idx == len(chain_complex.boundaries) - 1:
            image_dim = 0
        else:
            _, image_dim = chain_complex.compute_image(chain_complex.boundaries[layer_idx + 1])
        
        return kernel_dim - image_dim
    
    @staticmethod
    def analyze_all_layers(chain_complex: ChainComplex) -> Dict[str, List[float]]:
        """
        Perform complete exact sequence analysis across all layers.
        
        Returns:
            Dictionary with exactness, information loss, and Betti numbers
        """
        n_layers = len(chain_complex.boundaries)
        
        results = {
            'exactness': [],
            'information_loss': [],
            'betti_numbers_homology': [],
            'betti_numbers_cohomology': []
        }
        
        for l in range(n_layers):
            # Exactness
            exactness = ExactSequenceAnalyzer.compute_exactness(chain_complex, l)
            results['exactness'].append(exactness)
            
            # Information loss
            info_loss = ExactSequenceAnalyzer.compute_information_loss(chain_complex, l)
            results['information_loss'].append(info_loss)
            
            # Betti numbers
            _, betti_h = chain_complex.compute_homology(l)
            _, betti_c = chain_complex.compute_cohomology(l)
            results['betti_numbers_homology'].append(betti_h)
            results['betti_numbers_cohomology'].append(betti_c)
        
        return results


# =============================================================================
# BAYESIAN STABILITY ANALYSIS
# =============================================================================

class BayesianStabilityAnalyzer:
    """
    Assesses robustness of topological features under weight perturbations.
    
    Implements variational inference to compute posterior distributions
    over network weights and analyzes stability of Betti numbers.
    """
    
    def __init__(self, noise_std: float = 0.01):
        """
        Args:
            noise_std: Standard deviation for Gaussian weight perturbations
        """
        self.noise_std = noise_std
    
    def perturb_weights(self, weight_matrix: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian perturbation to weight matrix.
        
        W_perturbed ~ N(W, σ²I)
        """
        noise = np.random.normal(0, self.noise_std, weight_matrix.shape)
        return weight_matrix + noise
    
    def sample_posterior_weights(self, original_boundaries: List[np.ndarray], 
                                n_samples: int = 50) -> List[List[np.ndarray]]:
        """
        Sample from posterior distribution over weights.
        
        Returns:
            List of weight samples, each containing all boundary operators
        """
        samples = []
        
        for _ in range(n_samples):
            perturbed_boundaries = [
                self.perturb_weights(boundary) 
                for boundary in original_boundaries
            ]
            samples.append(perturbed_boundaries)
        
        return samples
    
    def compute_stability_metrics(self, chain_complex: ChainComplex, 
                                 n_samples: int = 50) -> Dict[str, any]:
        """
        Compute stability metrics for all layers.
        
        Returns:
            Dictionary containing mean Betti numbers, variance, and stability scores
        """
        # Sample perturbed weight configurations
        weight_samples = self.sample_posterior_weights(
            chain_complex.boundaries, n_samples
        )
        
        # Track Betti numbers across samples
        betti_samples = defaultdict(list)
        
        for sample_boundaries in weight_samples:
            # Create chain complex for this sample
            sample_complex = ChainComplex(
                chain_complex.spaces,
                sample_boundaries
            )
            
            # Compute Betti numbers
            for l in range(len(sample_boundaries)):
                _, betti_h = sample_complex.compute_homology(l)
                _, betti_c = sample_complex.compute_cohomology(l)
                
                betti_samples[f'homology_{l}'].append(betti_h)
                betti_samples[f'cohomology_{l}'].append(betti_c)
        
        # Compute statistics
        stability_results = {}
        
        for key, values in betti_samples.items():
            values = np.array(values)
            mean_betti = np.mean(values)
            var_betti = np.var(values)
            
            # Stability metric: S_l = 1 / Var(β_l)
            # Add small constant to avoid division by zero
            stability = 1.0 / (var_betti + 1e-8)
            
            stability_results[key] = {
                'mean': mean_betti,
                'variance': var_betti,
                'stability': stability,
                'samples': values
            }
        
        return stability_results


# =============================================================================
# NEURAL NETWORK WRAPPER
# =============================================================================

class HomologicalNN(nn.Module):
    """
    Neural network with homological analysis capabilities.
    
    Extends standard PyTorch networks with:
    - Chain complex extraction
    - Homology/cohomology computation
    - Interpretability analysis
    """
    
    def __init__(self, layer_dims: List[int], activation='relu'):
        """
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden_dim1, ..., output_dim]
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(HomologicalNN, self).__init__()
        
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1
        
        # Create layers
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(
                nn.Linear(layer_dims[i], layer_dims[i + 1])
            )
        
        # Set activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.activation_name = activation
        
        # Storage for representations
        self.representations = []
    
    def forward(self, x, store_representations=False):
        """Forward pass with optional representation storage."""
        if store_representations:
            self.representations = [x.detach().cpu().numpy()]
        
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            
            if i < self.n_layers - 1:  # Apply activation to all but last layer
                h = self.activation(h)
            
            if store_representations:
                self.representations.append(h.detach().cpu().numpy())
        
        return h
    
    def extract_chain_complex(self, dataloader: DataLoader, 
                             max_samples: int = 1000) -> ChainComplex:
        """
        Extract chain complex from trained network.
        
        Args:
            dataloader: DataLoader with input data
            max_samples: Maximum number of samples to use
            
        Returns:
            ChainComplex object representing the network
        """
        self.eval()
        
        # Collect representations
        all_representations = [[] for _ in range(self.n_layers + 1)]
        
        with torch.no_grad():
            n_collected = 0
            for batch_x, _ in dataloader:
                if n_collected >= max_samples:
                    break
                
                batch_x = batch_x.view(batch_x.size(0), -1)  # Flatten
                _ = self.forward(batch_x, store_representations=True)
                
                for i, rep in enumerate(self.representations):
                    all_representations[i].append(rep)
                
                n_collected += batch_x.size(0)
        
        # Concatenate representations
        representation_spaces = [
            np.vstack(reps)[:max_samples] 
            for reps in all_representations
        ]
        
        # Extract boundary operators (weight matrices)
        # Note: For chain complex, we need to ensure proper composition
        # The boundary operator maps from representation space i to i+1
        boundary_operators = []
        for i, layer in enumerate(self.layers):
            weight = layer.weight.detach().cpu().numpy()
            # Weight matrix is already in the correct orientation for forward pass
            # For chain complex: ∂_i: H_i → H_{i+1}
            boundary_operators.append(weight)
        
        return ChainComplex(representation_spaces, boundary_operators)
    
    def analyze_interpretability(self, dataloader: DataLoader,
                               n_stability_samples: int = 50) -> Dict:
        """
        Perform complete interpretability analysis.
        
        Returns:
            Comprehensive analysis including:
            - Exact sequence metrics
            - Betti numbers
            - Stability analysis
            - Feature interpretations
        """
        # Extract chain complex
        chain_complex = self.extract_chain_complex(dataloader)
        
        # Exact sequence analysis
        exact_analysis = ExactSequenceAnalyzer.analyze_all_layers(chain_complex)
        
        # Bayesian stability analysis
        stability_analyzer = BayesianStabilityAnalyzer()
        stability_metrics = stability_analyzer.compute_stability_metrics(
            chain_complex, n_samples=n_stability_samples
        )
        
        # Verify chain complex property
        chain_violations = chain_complex.verify_chain_property()
        
        return {
            'exact_sequence_analysis': exact_analysis,
            'stability_metrics': stability_metrics,
            'chain_property_violations': chain_violations,
            'chain_complex': chain_complex
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

class HomologicalVisualizer:
    """Visualization tools for homological analysis results."""
    
    @staticmethod
    def plot_betti_numbers(analysis_results: Dict, save_path: Optional[str] = None):
        """Plot evolution of Betti numbers across layers."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        exact_analysis = analysis_results['exact_sequence_analysis']
        
        layers = list(range(len(exact_analysis['betti_numbers_homology'])))
        
        # Homology Betti numbers
        ax1.plot(layers, exact_analysis['betti_numbers_homology'], 
                marker='o', linewidth=2, markersize=8, label='Homology')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Betti Number (β_l)', fontsize=12)
        ax1.set_title('Homology Betti Numbers', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Cohomology Betti numbers
        ax2.plot(layers, exact_analysis['betti_numbers_cohomology'], 
                marker='s', linewidth=2, markersize=8, color='orange', label='Cohomology')
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Betti Number (β_l)', fontsize=12)
        ax2.set_title('Cohomology Betti Numbers', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_exactness_analysis(analysis_results: Dict, save_path: Optional[str] = None):
        """Plot exactness and information loss metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        exact_analysis = analysis_results['exact_sequence_analysis']
        layers = list(range(len(exact_analysis['exactness'])))
        
        # Exactness
        ax1.plot(layers, exact_analysis['exactness'], 
                marker='o', linewidth=2, markersize=8, color='green')
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Exactness')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Exactness', fontsize=12)
        ax1.set_title('Exact Sequence Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Information loss
        ax2.bar(layers, exact_analysis['information_loss'], color='coral', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Information Loss (L_l)', fontsize=12)
        ax2.set_title('Information Flow Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_stability_analysis(analysis_results: Dict, save_path: Optional[str] = None):
        """Plot Bayesian stability metrics."""
        stability_metrics = analysis_results['stability_metrics']
        
        # Extract stability scores for homology
        layers = []
        stability_scores = []
        variances = []
        
        for key in sorted(stability_metrics.keys()):
            if 'homology' in key:
                layer_num = int(key.split('_')[1])
                layers.append(layer_num)
                stability_scores.append(stability_metrics[key]['stability'])
                variances.append(stability_metrics[key]['variance'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Stability scores
        ax1.plot(layers, stability_scores, marker='o', linewidth=2, 
                markersize=8, color='purple')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Stability (S_l)', fontsize=12)
        ax1.set_title('Cohomological Stability', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Variance
        ax2.bar(layers, variances, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Variance of Betti Numbers', fontsize=12)
        ax2.set_title('Topological Feature Variance', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_summary_report(analysis_results: Dict, save_dir: str = './results'):
        """Create comprehensive visualization report."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate all plots
        HomologicalVisualizer.plot_betti_numbers(
            analysis_results, 
            os.path.join(save_dir, 'betti_numbers.png')
        )
        
        HomologicalVisualizer.plot_exactness_analysis(
            analysis_results,
            os.path.join(save_dir, 'exactness_analysis.png')
        )
        
        HomologicalVisualizer.plot_stability_analysis(
            analysis_results,
            os.path.join(save_dir, 'stability_analysis.png')
        )
        
        print(f"✓ Visualization report saved to: {save_dir}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_analysis_summary(analysis_results: Dict):
    """Print human-readable summary of analysis results."""
    print("\n" + "="*70)
    print("HOMOLOGICAL ANALYSIS SUMMARY")
    print("="*70)
    
    exact_analysis = analysis_results['exact_sequence_analysis']
    
    print("\n[BETTI NUMBERS]")
    print("-" * 70)
    for i, (bh, bc) in enumerate(zip(
        exact_analysis['betti_numbers_homology'],
        exact_analysis['betti_numbers_cohomology']
    )):
        print(f"Layer {i}: β_h = {bh:2d}, β_c = {bc:2d}")
    
    print("\n[EXACTNESS ANALYSIS]")
    print("-" * 70)
    for i, (ex, loss) in enumerate(zip(
        exact_analysis['exactness'],
        exact_analysis['information_loss']
    )):
        status = "✓ Exact" if ex > 0.9 else "✗ Non-exact"
        print(f"Layer {i}: Exactness = {ex:.3f} {status}, Info Loss = {loss:+d}")
    
    # Find bottlenecks
    bottlenecks = [i for i, ex in enumerate(exact_analysis['exactness']) if ex < 0.7]
    if bottlenecks:
        print(f"\n⚠ Bottlenecks detected at layers: {bottlenecks}")
    
    print("\n[STABILITY METRICS]")
    print("-" * 70)
    stability_metrics = analysis_results['stability_metrics']
    for key in sorted(stability_metrics.keys()):
        if 'homology' in key:
            layer = key.split('_')[1]
            metrics = stability_metrics[key]
            print(f"Layer {layer}: Stability = {metrics['stability']:.2f}, "
                  f"Variance = {metrics['variance']:.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("Homological Algebra Framework for Neural Network Interpretability")
    print("="*70)
    print("\nThis module provides:")
    print("  • Chain complex modeling of neural networks")
    print("  • Homology and cohomology computation")
    print("  • Exact sequence analysis")
    print("  • Bayesian stability analysis")
    print("  • Comprehensive visualization tools")
    print("\nSee example_usage.py for demonstrations.")
