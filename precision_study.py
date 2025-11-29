"""
Machine Precision Analysis

Authors: Viki Mancoridis & Bella Stewart
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from bicgstab import bicgstab
from helpers import run_scipy_solver
from test_problems import poisson_2d, convection_diffusion_2d


def test_precision_levels(A, b, x_true, problem_name, precisions=None):
    """
    Test solver performance across different floating-point precisions
    """
    if precisions is None:
        precisions = {
            'float64': np.float64,
            'float32': np.float32,
            'float16': np.float16
        }
    
    print("\n" + "="*80)
    print(f"MACHINE PRECISION SWEEP: {problem_name}")
    print("="*80)
    print(f"Problem size: {A.shape[0]} unknowns")
    print(f"Testing precisions: {list(precisions.keys())}")
    print("="*80)
    
    results = {
        'precisions': list(precisions.keys()),
        'BiCGSTAB (Ours)': {
            'iterations': [],
            'errors': [],
            'converged': [],
            'times': [],
            'reasons': [],
            'final_residuals': []
        }
    }
    
    print(f"\n{'Precision':<12} {'Bits':<8} {'ε_machine':<15} {'Converged':<12} "
          f"{'Iters':<8} {'Error':<12} {'Reason':<20}")
    print("-"*95)
    
    for prec_name, dtype in precisions.items():
        # Machine epsilon for this precision
        eps_machine = np.finfo(dtype).eps
        
        # Determine bits
        if dtype == np.float16:
            bits = 16
        elif dtype == np.float32:
            bits = 32
        elif dtype == np.float64:
            bits = 64
        else:
            bits = '?'
        
        # Convert to specified precision
        try:
            A_prec = A.astype(dtype)
            b_prec = b.astype(dtype)
            x0_prec = np.zeros(b.shape[0], dtype=dtype)
            
            # Run solver
            t0 = time.time()
            x_prec, info = bicgstab(A_prec, b_prec, x0=x0_prec,
                                   tol=1e-6, maxiter=1000)
            elapsed = time.time() - t0
            
            # Compute error in float64 for fair comparison
            x_float64 = x_prec.astype(np.float64)
            error = np.linalg.norm(x_float64 - x_true)
            
            # Final residual
            r_final = info['residuals'][-1] if info['residuals'] else np.inf
            
            # Store results
            results['BiCGSTAB (Ours)']['iterations'].append(info['iterations'])
            results['BiCGSTAB (Ours)']['errors'].append(error)
            results['BiCGSTAB (Ours)']['converged'].append(info['converged'])
            results['BiCGSTAB (Ours)']['times'].append(elapsed)
            results['BiCGSTAB (Ours)']['reasons'].append(info['reason'])
            results['BiCGSTAB (Ours)']['final_residuals'].append(r_final)
            
            # Print results
            status = "✓ Yes" if info['converged'] else "✗ No"
            reason_short = info['reason'][:18] if len(info['reason']) > 18 else info['reason']
            
            print(f"{prec_name:<12} {bits:<8} {eps_machine:<15.2e} {status:<12} "
                  f"{info['iterations']:<8} {error:<12.3e} {reason_short:<20}")
            
        except Exception as e:
            # Handle catastrophic failure
            results['BiCGSTAB (Ours)']['iterations'].append(None)
            results['BiCGSTAB (Ours)']['errors'].append(None)
            results['BiCGSTAB (Ours)']['converged'].append(False)
            results['BiCGSTAB (Ours)']['times'].append(None)
            results['BiCGSTAB (Ours)']['reasons'].append(f'Exception: {str(e)[:30]}')
            results['BiCGSTAB (Ours)']['final_residuals'].append(None)
            
            print(f"{prec_name:<12} {bits:<8} {eps_machine:<15.2e} {'✗ CRASH':<12} "
                  f"{'-':<8} {'-':<12} {'Exception':<20}")
    
    print("="*95)
    
    return results


def test_precision_with_perturbations(A, b, x_true, problem_name,
                                      noise_levels=None):
    """
    Test sensitivity to rounding errors by adding controlled noise
    """
    if noise_levels is None:
        noise_levels = [0, 1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6]
    
    print("\n" + "="*80)
    print(f"NOISE PERTURBATION STUDY: {problem_name}")
    print("="*80)
    print("Adding controlled noise to RHS to simulate rounding errors")
    print("="*80)
    
    results = {
        'noise_levels': noise_levels,
        'iterations': [],
        'errors': [],
        'converged': [],
        'final_residuals': []
    }
    
    print(f"\n{'Noise Level':<15} {'Converged':<12} {'Iters':<8} "
          f"{'Error':<12} {'Final Residual':<15}")
    print("-"*70)
    
    for noise in noise_levels:
        # Add noise to RHS
        b_noisy = b + noise * np.linalg.norm(b) * np.random.randn(len(b))
        
        # Solve
        x, info = bicgstab(A, b_noisy, tol=1e-8, maxiter=1000)
        
        # Compute error (relative to perturbed problem's true solution)
        # For perturbed RHS, true solution is approximately x_true
        error = np.linalg.norm(x - x_true)
        
        final_res = info['residuals'][-1] if info['residuals'] else np.inf
        
        # Store results
        results['iterations'].append(info['iterations'])
        results['errors'].append(error)
        results['converged'].append(info['converged'])
        results['final_residuals'].append(final_res)
        
        # Print
        status = "✓ Yes" if info['converged'] else "✗ No"
        print(f"{noise:<15.2e} {status:<12} {info['iterations']:<8} "
              f"{error:<12.3e} {final_res:<15.3e}")
    
    print("="*70)
    
    return results


def test_breakdown_susceptibility_by_precision(A, b, x_true, problem_name,
                                               n_trials=50):
    """
    Test how precision affects breakdown frequency
    """
    print("\n" + "="*80)
    print(f"BREAKDOWN SUSCEPTIBILITY BY PRECISION: {problem_name}")
    print("="*80)
    print(f"Running {n_trials} trials with random initial guesses per precision")
    print("="*80)
    
    precisions = {
        'float64': np.float64,
        'float32': np.float32,
        'float16': np.float16
    }
    
    breakdown_stats = {}
    
    for prec_name, dtype in precisions.items():
        print(f"\nTesting {prec_name}...")
        
        breakdown_types = {
            'converged': 0,
            'rho breakdown': 0,
            'alpha breakdown': 0,
            'omega breakdown': 0,
            'maxiter': 0,
            'other': 0,
            'exception': 0
        }
        
        iteration_counts = []
        errors = []
        
        A_prec = A.astype(dtype)
        b_prec = b.astype(dtype)
        
        for trial in range(n_trials):
            # Random initial guess
            np.random.seed(trial)  # Reproducible
            x0 = np.random.randn(len(b)).astype(dtype)
            
            try:
                x, info = bicgstab(A_prec, b_prec, x0=x0, tol=1e-6, maxiter=1000)
                
                # Categorize result
                if info['converged']:
                    breakdown_types['converged'] += 1
                    iteration_counts.append(info['iterations'])
                    error = np.linalg.norm(x.astype(np.float64) - x_true)
                    errors.append(error)
                else:
                    reason = info['reason']
                    if 'rho' in reason:
                        breakdown_types['rho breakdown'] += 1
                    elif 'alpha' in reason:
                        breakdown_types['alpha breakdown'] += 1
                    elif 'omega' in reason:
                        breakdown_types['omega breakdown'] += 1
                    elif 'maxiter' in reason:
                        breakdown_types['maxiter'] += 1
                    else:
                        breakdown_types['other'] += 1
                        
            except Exception as e:
                breakdown_types['exception'] += 1
        
        # Compute statistics
        breakdown_stats[prec_name] = {
            'breakdown_types': breakdown_types,
            'success_rate': breakdown_types['converged'] / n_trials,
            'avg_iterations': np.mean(iteration_counts) if iteration_counts else None,
            'avg_error': np.mean(errors) if errors else None,
            'total_breakdowns': n_trials - breakdown_types['converged']
        }
    
    # Print summary
    print("\n" + "="*80)
    print("BREAKDOWN STATISTICS SUMMARY")
    print("="*80)
    print(f"\n{'Precision':<12} {'Success Rate':<15} {'Avg Iters':<12} "
          f"{'Breakdowns':<12} {'Exceptions':<12}")
    print("-"*70)
    
    for prec_name, stats in breakdown_stats.items():
        success_pct = stats['success_rate'] * 100
        avg_iters = f"{stats['avg_iterations']:.1f}" if stats['avg_iterations'] else "N/A"
        breakdowns = stats['total_breakdowns']
        exceptions = stats['breakdown_types']['exception']
        
        print(f"{prec_name:<12} {success_pct:>6.1f}%{'':<8} {avg_iters:<12} "
              f"{breakdowns:<12} {exceptions:<12}")
    
    print("\n" + "-"*80)
    print("Breakdown Type Distribution:")
    print("-"*80)
    print(f"{'Precision':<12} {'Rho':<8} {'Alpha':<8} {'Omega':<8} "
          f"{'MaxIter':<10} {'Other':<8}")
    print("-"*70)
    
    for prec_name, stats in breakdown_stats.items():
        bt = stats['breakdown_types']
        print(f"{prec_name:<12} {bt['rho breakdown']:<8} {bt['alpha breakdown']:<8} "
              f"{bt['omega breakdown']:<8} {bt['maxiter']:<10} {bt['other']:<8}")
    
    print("="*80)
    
    return breakdown_stats


def plot_precision_comparison(results_list, problem_names):
    """
    Visualize precision effects across multiple problems
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    precisions_order = ['float64', 'float32', 'float16']
    colors_prec = {'float64': 'blue', 'float32': 'green', 'float16': 'red'}
    
    # ========================================================================
    # Plot 1: Iterations by Precision and Problem
    # ========================================================================
    ax = axes[0, 0]
    
    x_pos = np.arange(len(problem_names))
    width = 0.25
    
    for i, prec in enumerate(precisions_order):
        iters_list = []
        for results in results_list:
            idx = results['precisions'].index(prec)
            iters = results['BiCGSTAB (Ours)']['iterations'][idx]
            # Set failed to a high value for visibility
            iters_list.append(iters if iters is not None else 1000)
        
        ax.bar(x_pos + i*width, iters_list, width, label=prec,
               color=colors_prec[prec], alpha=0.7)
    
    ax.set_ylabel('Iterations', fontsize=13, fontweight='bold')
    ax.set_title('Iterations by Precision', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(problem_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, title='Precision')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Plot 2: Error by Precision
    # ========================================================================
    ax = axes[0, 1]
    
    for i, prec in enumerate(precisions_order):
        errors_list = []
        valid_problems = []
        
        for j, results in enumerate(results_list):
            idx = results['precisions'].index(prec)
            error = results['BiCGSTAB (Ours)']['errors'][idx]
            if error is not None:
                errors_list.append(error)
                valid_problems.append(j)
        
        if errors_list:
            ax.semilogy(valid_problems, errors_list, 'o-', 
                       label=prec, color=colors_prec[prec],
                       linewidth=2.5, markersize=8)
    
    ax.set_xlabel('Problem Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Solution Error ||x - x_true||', fontsize=13, fontweight='bold')
    ax.set_title('Accuracy by Precision', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, title='Precision')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(problem_names)))
    ax.set_xticklabels([f'P{i}' for i in range(len(problem_names))], fontsize=10)
    
    # ========================================================================
    # Plot 3: Success Rate by Precision
    # ========================================================================
    ax = axes[1, 0]
    
    success_rates = {prec: [] for prec in precisions_order}
    
    for results in results_list:
        for prec in precisions_order:
            idx = results['precisions'].index(prec)
            converged = results['BiCGSTAB (Ours)']['converged'][idx]
            success_rates[prec].append(1 if converged else 0)
    
    x_pos = np.arange(len(precisions_order))
    overall_success = [100 * np.mean(success_rates[prec]) for prec in precisions_order]
    
    bars = ax.bar(x_pos, overall_success, color=[colors_prec[p] for p in precisions_order],
                  alpha=0.7)
    
    ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Overall Success Rate', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(precisions_order, fontsize=12)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rate in zip(bars, overall_success):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{rate:.0f}%', ha='center', va='bottom', 
               fontsize=12, fontweight='bold')
    
    # ========================================================================
    # Plot 4: Precision vs Machine Epsilon
    # ========================================================================
    ax = axes[1, 1]
    
    # Show relationship between machine epsilon and convergence
    eps_machine = {
        'float64': np.finfo(np.float64).eps,
        'float32': np.finfo(np.float32).eps,
        'float16': np.finfo(np.float16).eps
    }
    
    avg_iters = []
    for prec in precisions_order:
        iters_all = []
        for results in results_list:
            idx = results['precisions'].index(prec)
            iters = results['BiCGSTAB (Ours)']['iterations'][idx]
            if iters is not None:
                iters_all.append(iters)
        avg_iters.append(np.mean(iters_all) if iters_all else 0)
    
    eps_vals = [eps_machine[p] for p in precisions_order]
    
    ax.loglog(eps_vals, avg_iters, 'o-', linewidth=3, markersize=12,
             color='purple')
    
    for i, (eps, iters, prec) in enumerate(zip(eps_vals, avg_iters, precisions_order)):
        ax.annotate(prec, (eps, iters), textcoords="offset points",
                   xytext=(10, -5), fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Machine Epsilon ε', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Iterations', fontsize=13, fontweight='bold')
    ax.set_title('Precision vs Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Machine Precision Impact on BiCGSTAB Performance',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def plot_noise_sensitivity(results_noise, problem_name):
    """
    Visualize sensitivity to numerical noise
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    noise_levels = results_noise['noise_levels']
    iterations = results_noise['iterations']
    errors = results_noise['errors']
    
    # Plot 1: Iterations vs Noise
    ax = axes[0]
    ax.semilogx(noise_levels[1:], iterations[1:], 'o-', 
               linewidth=2.5, markersize=8, color='blue')
    ax.axhline(iterations[0], color='red', linestyle='--', 
              linewidth=2, label=f'No noise: {iterations[0]} iters')
    ax.set_xlabel('Relative Noise Level', fontsize=13, fontweight='bold')
    ax.set_ylabel('Iterations to Convergence', fontsize=13, fontweight='bold')
    ax.set_title('Iteration Count vs Noise', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error vs Noise
    ax = axes[1]
    ax.loglog(noise_levels[1:], errors[1:], 's-',
             linewidth=2.5, markersize=8, color='green')
    ax.axhline(errors[0], color='red', linestyle='--',
              linewidth=2, label=f'No noise: {errors[0]:.2e}')
    ax.set_xlabel('Relative Noise Level', fontsize=13, fontweight='bold')
    ax.set_ylabel('Solution Error', fontsize=13, fontweight='bold')
    ax.set_title('Accuracy vs Noise', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle(f'Noise Sensitivity: {problem_name}',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_breakdown_by_precision(breakdown_stats, problem_name):
    """
    Visualize breakdown statistics by precision
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    precisions = list(breakdown_stats.keys())
    
    # Plot 1: Success Rate
    ax = axes[0]
    success_rates = [breakdown_stats[p]['success_rate'] * 100 for p in precisions]
    colors = ['blue', 'green', 'red']
    
    bars = ax.bar(precisions, success_rates, color=colors, alpha=0.7)
    ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Convergence Success Rate', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{rate:.1f}%', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    # Plot 2: Breakdown Type Distribution
    ax = axes[1]
    
    breakdown_types = ['rho breakdown', 'alpha breakdown', 'omega breakdown',
                      'maxiter', 'exception']
    colors_breakdown = ['#ff9999', '#ffcc99', '#ffff99', '#99ccff', '#ff99ff']
    
    x = np.arange(len(precisions))
    width = 0.15
    
    for i, bd_type in enumerate(breakdown_types):
        counts = [breakdown_stats[p]['breakdown_types'][bd_type] for p in precisions]
        ax.bar(x + i*width, counts, width, label=bd_type.replace(' breakdown', ''),
              color=colors_breakdown[i], alpha=0.7)
    
    ax.set_ylabel('Number of Occurrences', fontsize=13, fontweight='bold')
    ax.set_title('Breakdown Type Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(precisions, fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Breakdown Analysis by Precision: {problem_name}',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


def analyze_precision_requirements(results_list, problem_names):
    """
    Analyze minimum precision requirements for different problems
    """
    print("\n" + "="*80)
    print("PRECISION REQUIREMENTS ANALYSIS")
    print("="*80)
    
    print(f"\n{'Problem':<30} {'float16':<15} {'float32':<15} {'float64':<15}")
    print("-"*75)
    
    for prob_name, results in zip(problem_names, results_list):
        statuses = []
        
        for prec in ['float16', 'float32', 'float64']:
            idx = results['precisions'].index(prec)
            converged = results['BiCGSTAB (Ours)']['converged'][idx]
            statuses.append("✓ OK" if converged else "✗ FAIL")
        
        print(f"{prob_name:<30} {statuses[0]:<15} {statuses[1]:<15} {statuses[2]:<15}")