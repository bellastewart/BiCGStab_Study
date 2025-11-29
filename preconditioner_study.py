"""
Preconditioner Strength Analysis

Study how preconditioner parameters affect solver performance

Authors: Viki Mancoridis & Bella Stewart
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from bicgstab import bicgstab
from helpers import build_ilu_preconditioner


def sweep_drop_tolerance(A, b, x_true, problem_name="Test Problem"):
    """
    Sweep drop tolerance from weak to strong preconditioning
    """
    print("\n" + "="*80)
    print(f"DROP TOLERANCE SWEEP: {problem_name}")
    print("="*80)
    print("Testing preconditioner strength by varying drop_tol...")
    print("Lower drop_tol → stronger preconditioner (keeps more entries)\n")
    
    # Range of drop tolerances
    drop_tols = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    fill_factor = 2.0  # Fixed fill factor
    
    results = {
        'drop_tols': drop_tols,
        'iterations': [],
        'errors': [],
        'times': [],
        'setup_times': [],
        'nnz_factors': [],
        'converged': [],
        'speedup': []
    }
    
    # Baseline: no preconditioner
    print(f"{'Drop Tol':<12} {'Setup(s)':<10} {'Iters':<8} {'Time(s)':<10} "
          f"{'NNZ':<10} {'Error':<12} {'Status':<10}")
    print("-"*80)
    
    print(f"{'None':<12} {'-':<10} ", end='')
    t0 = time.time()
    x_base, info_base = bicgstab(A, b, tol=1e-8, maxiter=1000, M=None)
    t_base = time.time() - t0
    err_base = np.linalg.norm(x_base - x_true)
    
    baseline_iters = info_base['iterations']
    baseline_time = t_base
    
    print(f"{baseline_iters:<8} {t_base:<10.4f} {'-':<10} "
          f"{err_base:<12.3e} {'✓' if info_base['converged'] else '✗':<10}")
    
    # Sweep drop tolerances
    for drop_tol in drop_tols:
        # Build preconditioner
        t_setup_start = time.time()
        M, success, nnz = build_ilu_preconditioner(A, 
                                                   fill_factor=fill_factor,
                                                   drop_tol=drop_tol)
        t_setup = time.time() - t_setup_start
        
        if not success:
            print(f"{drop_tol:<12.0e} {'FAILED':<10} {'-':<8} {'-':<10} "
                  f"{'-':<10} {'-':<12} {'✗':<10}")
            results['iterations'].append(None)
            results['errors'].append(None)
            results['times'].append(None)
            results['setup_times'].append(None)
            results['nnz_factors'].append(None)
            results['converged'].append(False)
            results['speedup'].append(None)
            continue
        
        # Solve with preconditioner
        t0 = time.time()
        x, info = bicgstab(A, b, tol=1e-8, maxiter=1000, M=M)
        t_solve = time.time() - t0
        
        err = np.linalg.norm(x - x_true)
        converged = info['converged']
        iters = info['iterations']
        
        # Compute speedup (iteration reduction)
        speedup = baseline_iters / iters if converged else None
        
        # Store results
        results['iterations'].append(iters)
        results['errors'].append(err)
        results['times'].append(t_solve)
        results['setup_times'].append(t_setup)
        results['nnz_factors'].append(nnz)
        results['converged'].append(converged)
        results['speedup'].append(speedup)
        
        print(f"{drop_tol:<12.0e} {t_setup:<10.4f} {iters:<8} {t_solve:<10.4f} "
              f"{nnz:<10} {err:<12.3e} {'✓' if converged else '✗':<10}")
    
    print("-"*80)
    print(f"Baseline (no precond): {baseline_iters} iterations in {baseline_time:.4f}s")
    print("="*80)
    
    return results, baseline_iters, baseline_time


def sweep_fill_factor(A, b, x_true, problem_name="Test Problem"):
    """
    Sweep fill factor from ILU(0) to more fill
    """
    print("\n" + "="*80)
    print(f"FILL FACTOR SWEEP: {problem_name}")
    print("="*80)
    print("Testing preconditioner strength by varying fill_factor...")
    print("Higher fill_factor → stronger preconditioner (more fill-in)\n")
    
    # Range of fill factors
    fill_factors = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    drop_tol = 1e-4  # Fixed drop tolerance
    
    results = {
        'fill_factors': fill_factors,
        'iterations': [],
        'errors': [],
        'times': [],
        'setup_times': [],
        'nnz_factors': [],
        'converged': [],
        'speedup': []
    }
    
    # Baseline: no preconditioner
    print(f"{'Fill':<12} {'Setup(s)':<10} {'Iters':<8} {'Time(s)':<10} "
          f"{'NNZ':<10} {'Error':<12} {'Status':<10}")
    print("-"*80)
    
    print(f"{'None':<12} {'-':<10} ", end='')
    t0 = time.time()
    x_base, info_base = bicgstab(A, b, tol=1e-8, maxiter=1000, M=None)
    t_base = time.time() - t0
    err_base = np.linalg.norm(x_base - x_true)
    
    baseline_iters = info_base['iterations']
    baseline_time = t_base
    
    print(f"{baseline_iters:<8} {t_base:<10.4f} {'-':<10} "
          f"{err_base:<12.3e} {'✓' if info_base['converged'] else '✗':<10}")
    
    # Sweep fill factors
    for fill in fill_factors:
        # Build preconditioner
        t_setup_start = time.time()
        M, success, nnz = build_ilu_preconditioner(A, 
                                                   fill_factor=fill,
                                                   drop_tol=drop_tol)
        t_setup = time.time() - t_setup_start
        
        if not success:
            print(f"{fill:<12.1f} {'FAILED':<10} {'-':<8} {'-':<10} "
                  f"{'-':<10} {'-':<12} {'✗':<10}")
            results['iterations'].append(None)
            results['errors'].append(None)
            results['times'].append(None)
            results['setup_times'].append(None)
            results['nnz_factors'].append(None)
            results['converged'].append(False)
            results['speedup'].append(None)
            continue
        
        # Solve with preconditioner
        t0 = time.time()
        x, info = bicgstab(A, b, tol=1e-8, maxiter=1000, M=M)
        t_solve = time.time() - t0
        
        err = np.linalg.norm(x - x_true)
        converged = info['converged']
        iters = info['iterations']
        
        # Compute speedup
        speedup = baseline_iters / iters if converged else None
        
        # Store results
        results['iterations'].append(iters)
        results['errors'].append(err)
        results['times'].append(t_solve)
        results['setup_times'].append(t_setup)
        results['nnz_factors'].append(nnz)
        results['converged'].append(converged)
        results['speedup'].append(speedup)
        
        print(f"{fill:<12.1f} {t_setup:<10.4f} {iters:<8} {t_solve:<10.4f} "
              f"{nnz:<10} {err:<12.3e} {'✓' if converged else '✗':<10}")
    
    print("-"*80)
    print(f"Baseline (no precond): {baseline_iters} iterations in {baseline_time:.4f}s")
    print("="*80)
    
    return results, baseline_iters, baseline_time


def plot_preconditioner_sweep(results_drop, results_fill, baseline_iters, 
                              problem_name="Test Problem"):
    """
    Visualize preconditioner strength sweep results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ========================================================================
    # Row 1: Drop Tolerance Sweep
    # ========================================================================
    
    # Filter out failed cases
    drop_tols = results_drop['drop_tols']
    iters_drop = [it for it in results_drop['iterations'] if it is not None]
    nnz_drop = [n for n in results_drop['nnz_factors'] if n is not None]
    times_drop = [t for t in results_drop['times'] if t is not None]
    speedup_drop = [s for s in results_drop['speedup'] if s is not None]
    valid_drop_tols = [dt for dt, it in zip(drop_tols, results_drop['iterations']) 
                       if it is not None]
    
    # Plot 1: Iterations vs Drop Tolerance
    ax = axes[0, 0]
    ax.semilogx(valid_drop_tols, iters_drop, 'o-', linewidth=2.5, 
                markersize=8, color='blue', label='With ILU')
    ax.axhline(baseline_iters, color='red', linestyle='--', linewidth=2,
               label='No Preconditioner')
    ax.set_xlabel('Drop Tolerance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Iterations to Convergence', fontsize=12, fontweight='bold')
    ax.set_title('Drop Tolerance vs Iterations', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Lower drop_tol = stronger
    
    # Plot 2: NNZ vs Drop Tolerance
    ax = axes[0, 1]
    ax.semilogx(valid_drop_tols, nnz_drop, 's-', linewidth=2.5, 
                markersize=8, color='green')
    ax.set_xlabel('Drop Tolerance', fontsize=12, fontweight='bold')
    ax.set_ylabel('NNZ in ILU Factors', fontsize=12, fontweight='bold')
    ax.set_title('Preconditioner Density', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Plot 3: Speedup vs Drop Tolerance
    ax = axes[0, 2]
    ax.semilogx(valid_drop_tols, speedup_drop, '^-', linewidth=2.5, 
                markersize=8, color='purple')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Drop Tolerance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Iteration Speedup', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Factor', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # ========================================================================
    # Row 2: Fill Factor Sweep
    # ========================================================================
    
    # Filter out failed cases
    fill_factors = results_fill['fill_factors']
    iters_fill = [it for it in results_fill['iterations'] if it is not None]
    nnz_fill = [n for n in results_fill['nnz_factors'] if n is not None]
    times_fill = [t for t in results_fill['times'] if t is not None]
    speedup_fill = [s for s in results_fill['speedup'] if s is not None]
    valid_fills = [ff for ff, it in zip(fill_factors, results_fill['iterations']) 
                   if it is not None]
    
    # Plot 4: Iterations vs Fill Factor
    ax = axes[1, 0]
    ax.plot(valid_fills, iters_fill, 'o-', linewidth=2.5, 
            markersize=8, color='blue', label='With ILU')
    ax.axhline(baseline_iters, color='red', linestyle='--', linewidth=2,
               label='No Preconditioner')
    ax.set_xlabel('Fill Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Iterations to Convergence', fontsize=12, fontweight='bold')
    ax.set_title('Fill Factor vs Iterations', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: NNZ vs Fill Factor
    ax = axes[1, 1]
    ax.plot(valid_fills, nnz_fill, 's-', linewidth=2.5, 
            markersize=8, color='green')
    ax.set_xlabel('Fill Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel('NNZ in ILU Factors', fontsize=12, fontweight='bold')
    ax.set_title('Preconditioner Density', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Speedup vs Fill Factor
    ax = axes[1, 2]
    ax.plot(valid_fills, speedup_fill, '^-', linewidth=2.5, 
            markersize=8, color='purple')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Fill Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Iteration Speedup', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Factor', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Preconditioner Strength Analysis: {problem_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def analyze_cost_benefit(results_drop, results_fill, baseline_time):
    """
    Analyze cost-benefit tradeoff of preconditioning
    
    Total time = setup time + solve time
    """
    print("\n" + "="*80)
    print("COST-BENEFIT ANALYSIS")
    print("="*80)
    print("\nDrop Tolerance Sweep:")
    print(f"{'Drop Tol':<12} {'Setup(s)':<10} {'Solve(s)':<10} {'Total(s)':<10} "
          f"{'Speedup':<10}")
    print("-"*80)
    
    for i, dt in enumerate(results_drop['drop_tols']):
        if results_drop['converged'][i]:
            setup = results_drop['setup_times'][i]
            solve = results_drop['times'][i]
            total = setup + solve
            speedup = baseline_time / total
            print(f"{dt:<12.0e} {setup:<10.4f} {solve:<10.4f} {total:<10.4f} "
                  f"{speedup:<10.2f}×")
    
    print(f"\nBaseline (no precond): {baseline_time:.4f}s total")
    
    print("\n" + "="*80)
    print("Fill Factor Sweep:")
    print(f"{'Fill':<12} {'Setup(s)':<10} {'Solve(s)':<10} {'Total(s)':<10} "
          f"{'Speedup':<10}")
    print("-"*80)
    
    for i, ff in enumerate(results_fill['fill_factors']):
        if results_fill['converged'][i]:
            setup = results_fill['setup_times'][i]
            solve = results_fill['times'][i]
            total = setup + solve
            speedup = baseline_time / total
            print(f"{ff:<12.1f} {setup:<10.4f} {solve:<10.4f} {total:<10.4f} "
                  f"{speedup:<10.2f}×")
    
    print(f"\nBaseline (no precond): {baseline_time:.4f}s total")
    print("="*80)


def run_preconditioner_study(A, b, x_true, problem_name="Test Problem"):
    """
    Complete preconditioner strength study
    """
    print("\n" + "█"*80)
    print("PRECONDITIONER STRENGTH STUDY")
    print("█"*80)
    print(f"\nProblem: {problem_name}")
    print(f"Size: {A.shape[0]} unknowns")
    print(f"NNZ: {A.nnz}")
    
    # Sweep drop tolerance
    results_drop, baseline_iters, baseline_time = sweep_drop_tolerance(
        A, b, x_true, problem_name)
    
    # Sweep fill factor
    results_fill, _, _ = sweep_fill_factor(A, b, x_true, problem_name)
    
    # Plot results
    plot_preconditioner_sweep(results_drop, results_fill, baseline_iters, 
                             problem_name)
    
    # Cost-benefit analysis
    analyze_cost_benefit(results_drop, results_fill, baseline_time)
    
    return results_drop, results_fill