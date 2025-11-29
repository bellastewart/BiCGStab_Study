# BiCGSTAB Implementation and Comparison Study

**Authors:** Viki Mancoridis & Bella Stewart  
**Institution:** MIT CEE 

## Overview

From-scratch implementation of the BiCGSTAB iterative solver with comprehensive comparison to BiCG, CGS, and GMRES for nonsymmetric sparse linear systems.

## Key Results

-  **74 iterations** vs 125 (BiCG), 280 (GMRES), 416 (CGS)
-  **627× more stable** than CGS (max spike: 83× vs 51,758×)
-  **5× less memory** than GMRES
-  **1.7× speedup** with ILU(0) preconditioning

## Getting Started
```bash
# Clone repository
git clone https://github.com/bellastewart/BiCGStab_Study.git

# Navigate to project
cd BiCGStab_Study
```

## Installation
```bash
# Create environment
conda create -n bicgstab-study python=3.11 numpy scipy matplotlib jupyter ipykernel -y

# Activate environment
conda activate bicgstab-study

# Register as Jupyter kernel
python -m ipykernel install --user --name bicgstab-study --display-name "BiCGSTAB Study"
```

## Usage
```bash
# Launch Jupyter (assuming you're already in project directory)
jupyter notebook
```

**In Jupyter:**
1. Open `main.ipynb` and Select kernel: **BiCGSTAB Study** to run main 
2. Open `summary.ipynb` and Select kernel: **BiCGSTAB Study** to run summary


## Project Structure
```
├── bicgstab.py          # BiCGSTAB implementation
├── comparison.py        # Solver comparison
├── flops_study.py       # Compare FLOPs of BiCGSTAB vs. Scipy solvers
├── helpers.py           # Helpers for analysis
├── main.ipynb           # Run experiments
├── peclet_study.py      # Vary Peclet numbers for problem difficulty
├── preconditioner_study # Assess effects of the preconditioner
├── README.md            # README for ease of project use
├── summary.ipynb        # View results of main experiments
├── summary.py           # Summary analysis
├── test_problems.py     # Test problems
├── visualization.py     # Plotting
```

## References

[1] Hestenes, Magnus R., and Eduard Stiefel. "Methods of conjugate gradients for solving linear systems." Journal of research of the National Bureau of Standards 49.6 (1952): 409-436.
[2] Sonneveld, Peter. "CGS, a fast Lanczos-type solver for nonsymmetric linear systems." SIAM journal on scientific and statistical computing 10.1 (1989): 36-52.
[3] Van der Vorst, Henk A. "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems." SIAM Journal on scientific and Statistical Computing 13.2 (1992): 631-644.
