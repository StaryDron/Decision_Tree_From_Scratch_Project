# ML-from-Scratch

This repository contains implementations of fundamental **machine learning algorithms**
from scratch using **NumPy**.  
The goal of this project is to deepen the understanding of ML internals and provide
clean, educational reference implementations.

## Implemented algorithms

- **Supervised learning**
  - Linear Regression
  - Logistic Regression
  - Decision Tree (CART)
  - Random Forest
  - k-Nearest Neighbors (kNN)
  - Naive Bayes

- **Unsupervised learning**
  - k-Means Clustering
  - Principal Component Analysis (PCA)

Each algorithm lives in the `myml/` package and has a corresponding Jupyter notebook
in `notebooks/` where it is:

1. Derived theoretically (short explanation),
2. Implemented step by step,
3. Tested on a real dataset,
4. Compared against the scikit-learn implementation.

## Repository structure

```text
myml/         # core implementations (NumPy only)
notebooks/    # experiments and comparisons vs scikit-learn
data/         # (optional) small datasets
images/       # figures used in README / notebooks
tests/        # (optional) unit tests
