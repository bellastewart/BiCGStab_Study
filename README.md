# BiCGSTAB Implementation and Comparison Study

**Authors:** Viki Mancoridis & Bella Stewart  
**Institution:** MIT DeCoDE Lab

## Overview

From-scratch implementation of the BiCGSTAB iterative solver with comprehensive comparison to BiCG, CGS, and GMRES for nonsymmetric sparse linear systems.

## Key Results

-  **74 iterations** vs 125 (BiCG), 280 (GMRES), 416 (CGS)
-  **627× more stable** than CGS (max spike: 83× vs 51,758×)
-  **5× less memory** than GMRES
-  **1.7× speedup** with ILU(0) preconditioning

## Installation
```bash
# Create environment
conda create -n bicgstab-study python=3.11 -y

# Activate
conda activate bicgstab-study

# Install dependencies
conda install numpy scipy matplotlib jupyter notebook ipykernel -y
```

## Usage
```bash
# Launch Jupyter
jupyter notebook

# Run experiments
# 1. Open and run main.ipynb
# 2. Open and run summary.ipynb for results
```

## Project Structure

Run main.ipynb first to collect results followed by summary.ipynb to get analysis. 

```
├── bicgstab.py          # Validated BiCGSTAB implementation
├── comparison.py        # Solver comparison framework
├── test_problems.py     # Poisson & convection-diffusion problems
├── visualization.py     # Convergence plotting
├── summary.py          # Analysis and tables
├── main.ipynb          # Run experiments
└── summary.ipynb       # View results
```

## References

[1] Hestenes, Magnus R., and Eduard Stiefel. "Methods of conjugate gradients for solving linear systems." Journal of research of the National Bureau of Standards 49.6 (1952): 409-436.
[2] Sonneveld, Peter. "CGS, a fast Lanczos-type solver for nonsymmetric linear systems." SIAM journal on scientific and statistical computing 10.1 (1989): 36-52.
[3] Van der Vorst, Henk A. "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems." SIAM Journal on scientific and Statistical Computing 13.2 (1992): 631-644.
