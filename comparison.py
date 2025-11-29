"""
Solver Comparison Functions

Authors: Viki Mancoridis & Bella Stewart
"""

import numpy as np
import time
from bicgstab import bicgstab
from helpers import run_scipy_solver, build_ilu_preconditioner

def compare_all_solvers(A, b, x_true, problem_name, use_precond=False):
    """
    Compare Our BiCGSTAB vs SciPy Reference Implementations
    """
    print(f"\n{'='*70}")
    print(f"Problem: {problem_name}")
    print(f"Size: {A.shape[0]} unknowns")
    print(f"Preconditioning: {'ILU(0)' if use_precond else 'None'}")
    print(f"{'='*70}\n")

    # Build preconditioner if requested
    M = None
    if use_precond:
        M, success, nnz = build_ilu_preconditioner(A)  # ← FIX: Now expects 3 values
        if not success:
            print("Falling back to no preconditioning\n")
            M = None
        else:
            print(f"ILU preconditioner built: {nnz:,} nonzeros in factors\n")

    x0 = np.zeros_like(b)
    results = {}

    # ========================================================================
    # Our BiCGSTAB Implementation (Validated)
    # ========================================================================
    print("Running BiCGSTAB (Our Implementation - Validated)...")
    t0 = time.time()
    x, info = bicgstab(A, b, x0=x0, tol=1e-8, maxiter=1000, M=M)
    elapsed = time.time() - t0
    err = np.linalg.norm(x - x_true)

    results['BiCGSTAB (Ours)'] = {
        'info': info,
        'error': err,
        'time': elapsed,
        'x': x,
        'source': 'Our implementation'
    }
    status = "✓" if info['converged'] else "✗"
    print(f"  {status} Converged: {info['converged']}, Iters: {info['iterations']}, "
        f"Error: {err:.3e}, Time: {elapsed:.3f}s")

    # ========================================================================
    # SciPy Reference Implementations
    # ========================================================================

    # BiCG
    print("\nRunning BiCG (SciPy Reference)...")
    x, info, elapsed = run_scipy_solver('bicg', A, b, x0, tol=1e-8, maxiter=1000, M=M)
    err = np.linalg.norm(x - x_true)
    results['BiCG (SciPy)'] = {
        'info': info,
        'error': err,
        'time': elapsed,
        'x': x,
        'source': 'SciPy reference'
    }
    status = "✓" if info['converged'] else "✗"
    print(f"  {status} Converged: {info['converged']}, Iters: {info['iterations']}, "
        f"Error: {err:.3e}, Time: {elapsed:.3f}s")

    # CGS
    print("\nRunning CGS (SciPy Reference)...")
    x, info, elapsed = run_scipy_solver('cgs', A, b, x0, tol=1e-8, maxiter=1000, M=M)
    err = np.linalg.norm(x - x_true)
    results['CGS (SciPy)'] = {
        'info': info,
        'error': err,
        'time': elapsed,
        'x': x,
        'source': 'SciPy reference'
    }
    status = "✓" if info['converged'] else "✗"
    print(f"  {status} Converged: {info['converged']}, Iters: {info['iterations']}, "
        f"Error: {err:.3e}, Time: {elapsed:.3f}s")


    # GMRES(20)
    print("\nRunning GMRES(20) (SciPy Reference)...")
    x, info, elapsed = run_scipy_solver('gmres', A, b, x0, tol=1e-8,
                                    maxiter=1000, restart=20, M=M)
    err = np.linalg.norm(x - x_true)
    results['GMRES(20) (SciPy)'] = {
        'info': info,
        'error': err,
        'time': elapsed,
        'x': x,
        'source': 'SciPy reference'
    }
    status = "✓" if info['converged'] else "✗"

    # Display with proper iteration info
    if 'cycles' in info:
        cycles = info['cycles']
        inner_iters = info['iterations']
        print(f"  {status} Converged: {info['converged']}, "
            f"Cycles: {cycles} (~{inner_iters} inner iters), "
            f"Error: {err:.3e}, Time: {elapsed:.3f}s")
    else:
        print(f"  {status} Converged: {info['converged']}, Iters: {info['iterations']}, "
            f"Error: {err:.3e}, Time: {elapsed:.3f}s")
        
    # CG (only for symmetric problems)
    if 'poisson' in problem_name.lower():
        print("\nRunning CG (SciPy Reference - Symmetric Baseline)...")
        x, info, elapsed = run_scipy_solver('cg', A, b, x0, tol=1e-8, maxiter=1000, M=M)
        err = np.linalg.norm(x - x_true)
        results['CG (SciPy)'] = {
            'info': info,
            'error': err,
            'time': elapsed,
            'x': x,
            'source': 'SciPy reference'
        }
        status = "✓" if info['converged'] else "✗"
        print(f"  {status} Converged: {info['converged']}, Iters: {info['iterations']}, "
            f"Error: {err:.3e}, Time: {elapsed:.3f}s")

    return results