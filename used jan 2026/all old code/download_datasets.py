"""
Dataset Downloader for Homological Framework
Downloads MNIST and CIFAR-10 datasets

This script downloads the datasets to ./data/ directory
Total download size: ~174 MB (MNIST: 11MB, CIFAR-10: 163MB)
"""

import os
import sys

def download_datasets():
    """Download MNIST and CIFAR-10 datasets."""
    
    print("="*70)
    print("DATASET DOWNLOADER - Homological Framework")
    print("="*70)
    
    # Check if torch/torchvision are installed
    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError:
        print("\n❌ Error: PyTorch/TorchVision not installed!")
        print("\nPlease install first:")
        print("  pip install torch torchvision")
        sys.exit(1)
    
    # Create data directory
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    print(f"\n📁 Data directory: {data_dir}")
    
    # Basic transform (just convert to tensor)
    transform = transforms.ToTensor()
    
    # =========================================================================
    # DOWNLOAD MNIST
    # =========================================================================
    print("\n" + "-"*70)
    print("[1/2] Downloading MNIST Dataset")
    print("-"*70)
    print("Size: ~11 MB")
    print("Description: Handwritten digits (0-9), 28×28 grayscale")
    print("Samples: 70,000 (60k train + 10k test)")
    
    try:
        # Download training set
        print("\nDownloading MNIST training set...")
        mnist_train = torchvision.datasets.MNIST(
            root=data_dir, 
            train=True, 
            download=True, 
            transform=transform
        )
        print(f"✓ Training set: {len(mnist_train)} samples")
        
        # Download test set
        print("\nDownloading MNIST test set...")
        mnist_test = torchvision.datasets.MNIST(
            root=data_dir, 
            train=False, 
            download=True, 
            transform=transform
        )
        print(f"✓ Test set: {len(mnist_test)} samples")
        
        print("\n✅ MNIST downloaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Error downloading MNIST: {e}")
        return False
    
    # =========================================================================
    # DOWNLOAD CIFAR-10
    # =========================================================================
    print("\n" + "-"*70)
    print("[2/2] Downloading CIFAR-10 Dataset")
    print("-"*70)
    print("Size: ~163 MB")
    print("Description: Natural images (10 classes), 32×32 RGB")
    print("Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")
    print("Samples: 60,000 (50k train + 10k test)")
    
    try:
        # Download training set
        print("\nDownloading CIFAR-10 training set...")
        cifar_train = torchvision.datasets.CIFAR10(
            root=data_dir, 
            train=True, 
            download=True, 
            transform=transform
        )
        print(f"✓ Training set: {len(cifar_train)} samples")
        
        # Download test set
        print("\nDownloading CIFAR-10 test set...")
        cifar_test = torchvision.datasets.CIFAR10(
            root=data_dir, 
            train=False, 
            download=True, 
            transform=transform
        )
        print(f"✓ Test set: {len(cifar_test)} samples")
        
        print("\n✅ CIFAR-10 downloaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Error downloading CIFAR-10: {e}")
        return False
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("✅ ALL DATASETS DOWNLOADED SUCCESSFULLY!")
    print("="*70)
    
    # Check disk space
    def get_dir_size(path):
        """Calculate directory size in MB."""
        total = 0
        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size(entry.path)
        except Exception:
            pass
        return total / (1024 * 1024)  # Convert to MB
    
    size_mb = get_dir_size(data_dir)
    
    print(f"\n📊 Summary:")
    print(f"  Location: {os.path.abspath(data_dir)}")
    print(f"  Total size: {size_mb:.1f} MB")
    print(f"  MNIST samples: {len(mnist_train) + len(mnist_test):,}")
    print(f"  CIFAR-10 samples: {len(cifar_train) + len(cifar_test):,}")
    
    print(f"\n📁 Directory structure:")
    print(f"  {data_dir}/")
    print(f"  ├── MNIST/")
    print(f"  │   ├── raw/")
    print(f"  │   └── processed/")
    print(f"  └── cifar-10-batches-py/")
    
    print("\n🎯 Next steps:")
    print("  1. Run: python test_installation.py")
    print("  2. Then: python example_usage.py")
    
    return True


def verify_datasets():
    """Verify that datasets are downloaded and accessible."""
    
    print("\n" + "="*70)
    print("VERIFYING DATASETS")
    print("="*70)
    
    try:
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
    except ImportError:
        print("❌ PyTorch/TorchVision not installed")
        return False
    
    transform = transforms.ToTensor()
    data_dir = './data'
    
    print("\n[1/4] Checking MNIST training set...")
    try:
        mnist_train = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=False, transform=transform
        )
        print(f"  ✓ Found {len(mnist_train)} samples")
    except:
        print("  ✗ Not found")
        return False
    
    print("\n[2/4] Checking MNIST test set...")
    try:
        mnist_test = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=False, transform=transform
        )
        print(f"  ✓ Found {len(mnist_test)} samples")
    except:
        print("  ✗ Not found")
        return False
    
    print("\n[3/4] Checking CIFAR-10 training set...")
    try:
        cifar_train = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=transform
        )
        print(f"  ✓ Found {len(cifar_train)} samples")
    except:
        print("  ✗ Not found")
        return False
    
    print("\n[4/4] Checking CIFAR-10 test set...")
    try:
        cifar_test = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=transform
        )
        print(f"  ✓ Found {len(cifar_test)} samples")
    except:
        print("  ✗ Not found")
        return False
    
    # Test data loading
    print("\n[5/5] Testing data loaders...")
    try:
        mnist_loader = DataLoader(mnist_test, batch_size=10, shuffle=False)
        cifar_loader = DataLoader(cifar_test, batch_size=10, shuffle=False)
        
        # Get one batch from each
        mnist_batch = next(iter(mnist_loader))
        cifar_batch = next(iter(cifar_loader))
        
        print(f"  ✓ MNIST batch shape: {mnist_batch[0].shape}")
        print(f"  ✓ CIFAR-10 batch shape: {cifar_batch[0].shape}")
        
    except Exception as e:
        print(f"  ✗ Data loader error: {e}")
        return False
    
    print("\n✅ All datasets verified and working!")
    return True


def show_sample_images():
    """Display sample images from both datasets."""
    
    print("\n" + "="*70)
    print("SAMPLE IMAGES (Optional Visualization)")
    print("="*70)
    
    try:
        import matplotlib.pyplot as plt
        import torchvision
        import torchvision.transforms as transforms
        import numpy as np
    except ImportError:
        print("⚠ Matplotlib not installed, skipping visualization")
        print("  Install with: pip install matplotlib")
        return
    
    transform = transforms.ToTensor()
    data_dir = './data'
    
    # Load datasets
    mnist = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=False, transform=transform
    )
    cifar = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform
    )
    
    # CIFAR-10 class names
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                     'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Images from Datasets', fontsize=16, fontweight='bold')
    
    # MNIST samples
    for i in range(5):
        img, label = mnist[i]
        axes[0, i].imshow(img.squeeze(), cmap='gray')
        axes[0, i].set_title(f'MNIST: {label}')
        axes[0, i].axis('off')
    
    # CIFAR-10 samples
    for i in range(5):
        img, label = cifar[i]
        # Convert from CxHxW to HxWxC
        img_np = img.permute(1, 2, 0).numpy()
        axes[1, i].imshow(img_np)
        axes[1, i].set_title(f'CIFAR-10: {cifar_classes[label]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = './data/sample_images.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Sample images saved to: {save_path}")
    
    # Try to display
    try:
        plt.show()
    except:
        print("  (Could not display - running in non-interactive mode)")


def main():
    """Main function."""
    
    # Check if data already exists
    data_dir = './data'
    mnist_exists = os.path.exists(os.path.join(data_dir, 'MNIST'))
    cifar_exists = os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))
    
    if mnist_exists and cifar_exists:
        print("\n" + "="*70)
        print("⚠️  DATASETS ALREADY EXIST")
        print("="*70)
        print(f"\nFound existing datasets in: {os.path.abspath(data_dir)}")
        
        choice = input("\nWhat would you like to do?\n"
                      "  1. Verify datasets (recommended)\n"
                      "  2. Re-download (will overwrite)\n"
                      "  3. Show sample images\n"
                      "  4. Exit\n"
                      "\nChoice (1-4): ").strip()
        
        if choice == '1':
            verify_datasets()
        elif choice == '2':
            download_datasets()
            verify_datasets()
        elif choice == '3':
            show_sample_images()
        else:
            print("\nExiting.")
            return
    else:
        # Download datasets
        success = download_datasets()
        
        if success:
            # Verify
            verify_datasets()
            
            # Ask about visualization
            choice = input("\nWould you like to see sample images? (y/n): ").lower()
            if choice == 'y':
                show_sample_images()


if __name__ == "__main__":
    main()
