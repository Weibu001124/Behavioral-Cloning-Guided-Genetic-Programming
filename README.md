# BCGP: Behavioral Cloning Guided Genetic Programming

This repository implements the **Behavioral Cloning Guided Genetic Programming (BCGP)** algorithm for **Symbolic Regression (SR)**.

BCGP improves crossover by preserving relationships between parent operators and subtrees, using a multilayer perceptron to guide subtree evolution.  

## Features
- Guided subtree crossover via behavioral cloning
- Supports multiple benchmark datasets
- Easy to configure parameters for experiments

## Requirements
- Python 3.8+
- NumPy, pandas
- Optional: PyTorch for neural network guidance
- Other dependencies listed in `environment.yml`

## Usage
```bash
# Activate environment
conda activate bcgp

# Run an example
python3 main.py 2025 1 70000000 500 1000
