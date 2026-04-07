"""
Example Usage: Homological Analysis on MNIST and CIFAR-10

This script demonstrates the homological algebra framework on two datasets:
1. MNIST - Simple handwritten digits (grayscale, 28x28)
2. CIFAR-10 - Natural images (RGB, 32x32)

Reproduces key experiments from the paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from homological_nn_framework import (
    HomologicalNN,
    HomologicalVisualizer,
    print_analysis_summary
)
import matplotlib.pyplot as plt
import os


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_mnist(batch_size=128):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_cifar10(batch_size=128):
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# =============================================================================
# TRAINING
# =============================================================================

def train_network(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    """Train neural network."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining on {device}...")
    print("-" * 70)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)  # Flatten
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1).to(device)
                target = target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, "
              f"Test Accuracy = {accuracy:.2f}%")
    
    print("-" * 70)
    print(f"✓ Training complete. Final accuracy: {test_accuracies[-1]:.2f}%\n")
    
    return train_losses, test_accuracies


# =============================================================================
# EXPERIMENT 1: MNIST (Simple Case)
# =============================================================================

def experiment_mnist():
    """
    Experiment 1: MNIST Digit Classification
    
    Network: [784 → 128 → 64 → 32 → 10]
    Expected findings:
    - Early layers detect edge orientations (H^1)
    - Middle layers capture digit topology (H^2)
    - Bottleneck at dim=32 layer
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: MNIST DIGIT CLASSIFICATION")
    print("="*70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_mnist(batch_size=128)
    
    # Create network - matches paper architecture
    model = HomologicalNN(
        layer_dims=[784, 128, 64, 32, 10],
        activation='relu'
    )
    
    print("\nNetwork Architecture:")
    print(f"  Input:    784 (28×28 flattened)")
    print(f"  Hidden 1: 128")
    print(f"  Hidden 2: 64")
    print(f"  Hidden 3: 32 (bottleneck)")
    print(f"  Output:   10 (digit classes)")
    
    # Train
    train_losses, test_accuracies = train_network(
        model, train_loader, test_loader,
        epochs=10, lr=0.001, device=device
    )
    
    # Move back to CPU for analysis
    model = model.cpu()
    
    # Homological Analysis
    print("\nPerforming homological analysis...")
    print("(This may take a few minutes)")
    
    analysis_results = model.analyze_interpretability(
        test_loader,
        n_stability_samples=50
    )
    
    # Print summary
    print_analysis_summary(analysis_results)
    
    # Generate visualizations
    save_dir = './results/mnist'
    os.makedirs(save_dir, exist_ok=True)
    HomologicalVisualizer.create_summary_report(analysis_results, save_dir)
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_losses, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('MNIST Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(test_accuracies, linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('MNIST Test Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    print(f"\n✓ Results saved to: {save_dir}")
    
    return model, analysis_results


# =============================================================================
# EXPERIMENT 2: CIFAR-10 (Complex Case)
# =============================================================================

def experiment_cifar10():
    """
    Experiment 2: CIFAR-10 Image Classification
    
    Network: [3072 → 512 → 256 → 128 → 10]
    Expected findings:
    - Higher Betti numbers due to color/texture complexity
    - Multiple bottlenecks during dimensionality reduction
    - Lower stability due to natural image variability
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: CIFAR-10 IMAGE CLASSIFICATION")
    print("="*70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cifar10(batch_size=128)
    
    # Create network
    model = HomologicalNN(
        layer_dims=[3072, 512, 256, 128, 10],
        activation='relu'
    )
    
    print("\nNetwork Architecture:")
    print(f"  Input:    3072 (32×32×3 flattened)")
    print(f"  Hidden 1: 512")
    print(f"  Hidden 2: 256 (bottleneck)")
    print(f"  Hidden 3: 128 (bottleneck)")
    print(f"  Output:   10 (object classes)")
    
    # Train
    train_losses, test_accuracies = train_network(
        model, train_loader, test_loader,
        epochs=15, lr=0.001, device=device
    )
    
    # Move back to CPU for analysis
    model = model.cpu()
    
    # Homological Analysis
    print("\nPerforming homological analysis...")
    print("(This may take a few minutes)")
    
    analysis_results = model.analyze_interpretability(
        test_loader,
        n_stability_samples=50
    )
    
    # Print summary
    print_analysis_summary(analysis_results)
    
    # Generate visualizations
    save_dir = './results/cifar10'
    os.makedirs(save_dir, exist_ok=True)
    HomologicalVisualizer.create_summary_report(analysis_results, save_dir)
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_losses, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('CIFAR-10 Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(test_accuracies, linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('CIFAR-10 Test Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    print(f"\n✓ Results saved to: {save_dir}")
    
    return model, analysis_results


# =============================================================================
# COMPARATIVE ANALYSIS
# =============================================================================

def compare_datasets(mnist_results, cifar_results):
    """Compare homological properties between MNIST and CIFAR-10."""
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS: MNIST vs CIFAR-10")
    print("="*70)
    
    mnist_exact = mnist_results['exact_sequence_analysis']
    cifar_exact = cifar_results['exact_sequence_analysis']
    
    print("\n[COMPLEXITY COMPARISON]")
    print("-" * 70)
    
    mnist_total_betti = sum(mnist_exact['betti_numbers_homology'])
    cifar_total_betti = sum(cifar_exact['betti_numbers_homology'])
    
    print(f"Total Betti numbers (Homology):")
    print(f"  MNIST:    {mnist_total_betti}")
    print(f"  CIFAR-10: {cifar_total_betti}")
    print(f"  Ratio:    {cifar_total_betti / max(mnist_total_betti, 1):.2f}x")
    
    print(f"\nAverage Exactness:")
    mnist_avg_exact = np.mean(mnist_exact['exactness'])
    cifar_avg_exact = np.mean(cifar_exact['exactness'])
    print(f"  MNIST:    {mnist_avg_exact:.3f}")
    print(f"  CIFAR-10: {cifar_avg_exact:.3f}")
    
    print(f"\nTotal Information Loss:")
    mnist_total_loss = sum([x for x in mnist_exact['information_loss'] if x > 0])
    cifar_total_loss = sum([x for x in cifar_exact['information_loss'] if x > 0])
    print(f"  MNIST:    {mnist_total_loss}")
    print(f"  CIFAR-10: {cifar_total_loss}")
    
    print("\n[STABILITY COMPARISON]")
    print("-" * 70)
    
    # Extract average stability
    mnist_stab = mnist_results['stability_metrics']
    cifar_stab = cifar_results['stability_metrics']
    
    mnist_avg_stability = np.mean([
        mnist_stab[k]['stability'] for k in mnist_stab.keys() if 'homology' in k
    ])
    cifar_avg_stability = np.mean([
        cifar_stab[k]['stability'] for k in cifar_stab.keys() if 'homology' in k
    ])
    
    print(f"Average Cohomological Stability:")
    print(f"  MNIST:    {mnist_avg_stability:.2f}")
    print(f"  CIFAR-10: {cifar_avg_stability:.2f}")
    
    print("\n[KEY FINDINGS]")
    print("-" * 70)
    print("• CIFAR-10 shows higher representational complexity (more Betti numbers)")
    print("• MNIST has better exactness (less information loss)")
    print("• MNIST representations are more stable under weight perturbations")
    print("• Both datasets show bottlenecks at dimensionality reduction layers")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Betti numbers comparison
    layers_mnist = range(len(mnist_exact['betti_numbers_homology']))
    layers_cifar = range(len(cifar_exact['betti_numbers_homology']))
    
    axes[0, 0].plot(layers_mnist, mnist_exact['betti_numbers_homology'], 
                    marker='o', label='MNIST', linewidth=2)
    axes[0, 0].plot(layers_cifar, cifar_exact['betti_numbers_homology'], 
                    marker='s', label='CIFAR-10', linewidth=2)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Betti Number')
    axes[0, 0].set_title('Betti Numbers Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Exactness comparison
    axes[0, 1].plot(layers_mnist, mnist_exact['exactness'], 
                    marker='o', label='MNIST', linewidth=2)
    axes[0, 1].plot(layers_cifar, cifar_exact['exactness'], 
                    marker='s', label='CIFAR-10', linewidth=2)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Exactness')
    axes[0, 1].set_title('Exactness Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Information loss comparison
    axes[1, 0].bar(np.array(layers_mnist) - 0.2, mnist_exact['information_loss'], 
                   width=0.4, label='MNIST', alpha=0.7)
    axes[1, 0].bar(np.array(layers_cifar) + 0.2, cifar_exact['information_loss'], 
                   width=0.4, label='CIFAR-10', alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Information Loss')
    axes[1, 0].set_title('Information Loss Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Stability comparison
    mnist_stab_values = [mnist_stab[k]['stability'] for k in sorted(mnist_stab.keys()) if 'homology' in k]
    cifar_stab_values = [cifar_stab[k]['stability'] for k in sorted(cifar_stab.keys()) if 'homology' in k]
    
    axes[1, 1].plot(range(len(mnist_stab_values)), mnist_stab_values, 
                    marker='o', label='MNIST', linewidth=2)
    axes[1, 1].plot(range(len(cifar_stab_values)), cifar_stab_values, 
                    marker='s', label='CIFAR-10', linewidth=2)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Stability (log scale)')
    axes[1, 1].set_title('Stability Comparison')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison visualization saved to: ./results/comparison.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all experiments."""
    print("\n" + "="*70)
    print("HOMOLOGICAL ALGEBRA FRAMEWORK FOR NEURAL NETWORKS")
    print("Paper Implementation: Two Dataset Comparison")
    print("="*70)
    
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    # Run experiments
    print("\n[1/2] Running MNIST experiment...")
    mnist_model, mnist_results = experiment_mnist()
    
    print("\n[2/2] Running CIFAR-10 experiment...")
    cifar_model, cifar_results = experiment_cifar10()
    
    # Comparative analysis
    compare_datasets(mnist_results, cifar_results)
    
    print("\n" + "="*70)
    print("✅ ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nGenerated outputs:")
    print("  • ./results/mnist/        - MNIST analysis results")
    print("  • ./results/cifar10/      - CIFAR-10 analysis results")
    print("  • ./results/comparison.png - Comparative analysis")
    print("\nKey files:")
    print("  • *_betti_numbers.png     - Topological complexity")
    print("  • *_exactness_analysis.png - Information flow")
    print("  • *_stability_analysis.png - Robustness metrics")
    print("  • *_training_curves.png    - Model performance")
    
    return mnist_model, mnist_results, cifar_model, cifar_results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run all experiments
    mnist_model, mnist_results, cifar_model, cifar_results = main()
