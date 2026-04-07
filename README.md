# Homological Algebra Framework for Neural Networks

## Overview
This framework applies homological algebra to analyze neural network representations, computing:
- **Betti numbers** (topological features per layer)
- **Exactness measures** (information flow quality)
- **Stability analysis** (robustness to perturbations)

## Datasets Included
- Toy Dataset (Circles vs Squares)
- Agriculture (UCI Wine - Precision Viticulture)
- MNIST (Handwritten Digits)
- CIFAR-10 (Object Recognition)
- Rotated MNIST (SO(2) Equivariance)
- Spherical Data (SO(3) Equivariance)

## Installation

```bash
# Clone repository
git clone https://github.com/SirajOmerNG/homological-algebra-framework.git
cd homological-algebra-framework

# Create conda environment
conda create -n homological python=3.9 -y
conda activate homological

# Install dependencies
pip install -r requirements.txt


# Homological Algebra Framework for Neural Networks

[![GitHub stars](https://img.shields.io/github/stars/SirajOmerNG/homological-algebra-framework)](https://github.com/SirajOmerNG/homological-algebra-framework/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This framework applies homological algebra to analyze neural network representations, computing:
- **Betti numbers** (topological features per layer)
- **Exactness measures** (information flow quality)
- **Stability analysis** (robustness to perturbations)

## Datasets Included

| Dataset | Description | Features | Classes |
|---------|-------------|----------|---------|
| Toy | Circles vs Squares (2D) | 2 | 2 |
| Agriculture | UCI Wine - Precision Viticulture | 13 | 3 |
| MNIST | Handwritten digits | 784 | 10 |
| CIFAR-10 | Object recognition | 3072 | 10 |
| Rotated MNIST | SO(2) equivariance test | 784 | 10 |
| Spherical | SO(3) equivariance test | 3 | 2 |

## Installation

### Using Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/SirajOmerNG/homological-algebra-framework.git
cd homological-algebra-framework

# Create conda environment
conda create -n homological python=3.9 -y
conda activate homological

# Install dependencies
pip install -r requirements.txt