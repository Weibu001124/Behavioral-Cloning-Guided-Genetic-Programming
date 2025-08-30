# BCGP: Behavioral Cloning Guided Genetic Programming

This repository implements the **Behavioral Cloning Guided Genetic Programming (BCGP)** algorithm for **Symbolic Regression (SR)**.

BCGP improves crossover by preserving relationships between parent operators and subtrees, using a multilayer perceptron to guide subtree evolution.  
It outperforms ellynGP and GP-GOMEA on benchmark problems from the SR benchmark and the Feyman SR database.

## Features
- Guided subtree crossover via behavioral cloning
- Supports multiple benchmark datasets
- Easy to configure parameters for experiments
- Log and JSON output for results analysis

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
python3 main.py f1 1000 100 50 0
