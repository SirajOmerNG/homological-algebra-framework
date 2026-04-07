"""
MASTER RUN SCRIPT - Complete Workflow
Runs everything in the correct order with progress tracking

Usage: python run_all.py
"""

import os
import sys
import time

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70)


def print_step(step_num, total_steps, description):
    """Print step information."""
    print(f"\n[STEP {step_num}/{total_steps}] {description}")
    print("-"*70)


def run_script(script_name, description):
    """Run a Python script and check for errors."""
    print(f"\nRunning: {script_name}")
    print(f"Purpose: {description}")
    
    start_time = time.time()
    
    try:
        # Import and run
        if script_name == "download_datasets.py":
            from download_datasets import download_datasets, verify_datasets
            success = download_datasets()
            if success:
                verify_datasets()
        
        elif script_name == "test_installation.py":
            import test_installation
        
        elif script_name == "example_usage.py":
            import example_usage
            example_usage.main()
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if all dependencies are installed."""
    print_header("CHECKING DEPENDENCIES")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing.append(module)
    
    if missing:
        print("\n❌ Missing dependencies!")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr use:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def estimate_time():
    """Estimate total time needed."""
    print_header("TIME ESTIMATE")
    
    print("\nEstimated time for complete workflow:")
    print("  • Dependency check:    < 1 minute")
    print("  • Download datasets:   2-5 minutes (depends on internet)")
    print("  • Test installation:   < 1 minute")
    print("  • MNIST experiment:    5-10 minutes")
    print("  • CIFAR-10 experiment: 10-15 minutes")
    print("  • Visualization:       < 1 minute")
    print("  " + "-"*60)
    print("  • TOTAL:               ~20-35 minutes")
    
    print("\n💡 Tips:")
    print("  • GPU will be faster (if available)")
    print("  • You can run individual experiments separately")
    print("  • Results will be saved to ./results/")


def show_options():
    """Show run options to user."""
    print_header("RUN OPTIONS")
    
    print("\nWhat would you like to do?")
    print("  1. Complete workflow (download + test + experiments)")
    print("  2. Download datasets only")
    print("  3. Test installation only")
    print("  4. Run experiments only (requires datasets)")
    print("  5. Run MNIST only (faster)")
    print("  6. Run CIFAR-10 only")
    print("  7. Show time estimates")
    print("  8. Exit")
    
    choice = input("\nChoice (1-8): ").strip()
    return choice


def run_complete_workflow():
    """Run the complete workflow."""
    print_header("COMPLETE WORKFLOW - HOMOLOGICAL FRAMEWORK")
    
    total_steps = 4
    
    # Step 1: Check dependencies
    print_step(1, total_steps, "Checking Dependencies")
    if not check_dependencies():
        print("\n❌ Cannot proceed without dependencies.")
        return False
    
    # Step 2: Download datasets
    print_step(2, total_steps, "Downloading Datasets (MNIST + CIFAR-10)")
    print("This will download ~174 MB of data")
    
    proceed = input("\nProceed with download? (y/n): ").lower()
    if proceed != 'y':
        print("Skipping download...")
    else:
        from download_datasets import download_datasets, verify_datasets
        success = download_datasets()
        if success:
            verify_datasets()
        else:
            print("\n❌ Dataset download failed.")
            return False
    
    # Step 3: Test installation
    print_step(3, total_steps, "Testing Installation")
    try:
        import test_installation
        print("\n✓ Installation test passed!")
    except Exception as e:
        print(f"\n✗ Installation test failed: {e}")
        return False
    
    # Step 4: Run experiments
    print_step(4, total_steps, "Running Experiments")
    print("\nThis will:")
    print("  • Train networks on MNIST and CIFAR-10")
    print("  • Perform homological analysis")
    print("  • Generate visualizations")
    print("  • Create comparison reports")
    
    proceed = input("\nProceed with experiments? (y/n): ").lower()
    if proceed != 'y':
        print("\nSkipping experiments.")
        print("You can run them later with: python example_usage.py")
        return True
    
    try:
        import example_usage
        example_usage.main()
        print("\n✓ Experiments completed!")
    except Exception as e:
        print(f"\n✗ Experiments failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Success summary
    print_header("✅ WORKFLOW COMPLETE!")
    
    print("\n📊 Results saved to:")
    print("  ./results/mnist/")
    print("  ./results/cifar10/")
    print("  ./results/comparison.png")
    
    print("\n📁 Generated files:")
    print("  • Betti numbers plots")
    print("  • Exactness analysis")
    print("  • Stability metrics")
    print("  • Training curves")
    print("  • Comparative analysis")
    
    print("\n🎯 Next steps:")
    print("  • Check ./results/ for all visualizations")
    print("  • Read README.md for interpretation guide")
    print("  • Modify example_usage.py for custom experiments")
    
    return True


def main():
    """Main entry point."""
    
    print_header("HOMOLOGICAL FRAMEWORK - MASTER RUNNER")
    print("\nThis script runs the complete workflow:")
    print("  1. Check dependencies")
    print("  2. Download datasets")
    print("  3. Test installation")
    print("  4. Run experiments")
    
    # Show options
    choice = show_options()
    
    if choice == '1':
        # Complete workflow
        run_complete_workflow()
    
    elif choice == '2':
        # Download only
        print_header("DOWNLOADING DATASETS")
        from download_datasets import download_datasets, verify_datasets
        success = download_datasets()
        if success:
            verify_datasets()
    
    elif choice == '3':
        # Test only
        print_header("TESTING INSTALLATION")
        import test_installation
    
    elif choice == '4':
        # Experiments only
        print_header("RUNNING EXPERIMENTS")
        import example_usage
        example_usage.main()
    
    elif choice == '5':
        # MNIST only
        print_header("MNIST EXPERIMENT ONLY")
        from example_usage import experiment_mnist
        model, results = experiment_mnist()
        print("\n✓ MNIST experiment complete!")
        print("  Results saved to: ./results/mnist/")
    
    elif choice == '6':
        # CIFAR-10 only
        print_header("CIFAR-10 EXPERIMENT ONLY")
        from example_usage import experiment_cifar10
        model, results = experiment_cifar10()
        print("\n✓ CIFAR-10 experiment complete!")
        print("  Results saved to: ./results/cifar10/")
    
    elif choice == '7':
        # Time estimates
        estimate_time()
    
    else:
        print("\nExiting.")
        return
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Set working directory to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run
    main()
