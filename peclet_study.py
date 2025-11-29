"""
Péclet Number Sweep Study

Authors: Viki Mancoridis & Bella Stewart
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from bicgstab import bicgstab
from helpers import run_scipy_solver, build_ilu_preconditioner
from test_problems import convection_diffusion_2d


def compute_peclet_number(vel, h, diffusion):
    """
    Compute Péclet number for convection-diffusion problem
    """
    vel_mag = np.sqrt(vel[0]**2 + vel[1]**2)
    pe = vel_mag * h / diffusion
    return pe


def sweep_peclet_numbers(nx=40, ny=40, diffusion=0.01, peclet_targets=None,
                        use_precond=False):
    """
    Test solver performance across range of Péclet numbers
    
    Strategy: Fix grid size and diffusion, vary velocity to hit target Péclet
    """
    if peclet_targets is None:
        peclet_targets = [0.1, 1, 5, 10, 25, 50, 100, 200, 500]
    
    h = 1.0 / (nx + 1)
    
    print("\n" + "="*80)
    print("PÉCLET NUMBER SWEEP")
    print("="*80)
    print(f"Grid: {nx}×{ny} ({nx*ny} unknowns)")
    print(f"Grid spacing h = {h:.4f}")
    print(f"Diffusion ε = {diffusion}")
    print(f"Preconditioning: {'ILU(0)' if use_precond else 'None'}")
    print(f"Target Péclet numbers: {peclet_targets}")
    print("="*80)
    
    # Storage for results
    results = {
        'peclet_numbers': [],
        'velocities': [],
        'BiCGSTAB (Ours)': {'iterations': [], 'times': [], 'errors': [], 'converged': []},
        'BiCG (SciPy)': {'iterations': [], 'times': [], 'errors': [], 'converged': []},
        'CGS (SciPy)': {'iterations': [], 'times': [], 'errors': [], 'converged': []},
        'GMRES(20) (SciPy)': {'iterations': [], 'times': [], 'errors': [], 'converged': []}
    }
    
    print(f"\n{'Pe':<8} {'Velocity':<12} {'BiCGSTAB':<12} {'BiCG':<12} "
          f"{'CGS':<12} {'GMRES':<12}")
    print("-"*80)
    
    for pe_target in peclet_targets:
        # Calculate velocity needed for target Péclet
        vel_mag = pe_target * diffusion / h
        # Use 45-degree flow direction
        vel = (vel_mag / np.sqrt(2), vel_mag / np.sqrt(2))
        
        # Build problem
        A = convection_diffusion_2d(nx, ny, diffusion=diffusion, vel=vel)
        N = A.shape[0]
        x_true = np.ones(N)
        b = A.dot(x_true)
        x0 = np.zeros_like(b)
        
        # Verify Péclet number
        pe_actual = compute_peclet_number(vel, h, diffusion)
        results['peclet_numbers'].append(pe_actual)
        results['velocities'].append(vel_mag)
        
        # Build preconditioner if requested
        M = None
        if use_precond:
            M, success, _ = build_ilu_preconditioner(A)
            if not success:
                print(f"Warning: ILU failed for Pe={pe_target}, using no precond")
                M = None
        
        # Test BiCGSTAB (ours)
        t0 = time.time()
        x, info = bicgstab(A, b, x0=x0, tol=1e-8, maxiter=1000, M=M)
        t_bicgstab = time.time() - t0
        err = np.linalg.norm(x - x_true)
        
        results['BiCGSTAB (Ours)']['iterations'].append(info['iterations'])
        results['BiCGSTAB (Ours)']['times'].append(t_bicgstab)
        results['BiCGSTAB (Ours)']['errors'].append(err)
        results['BiCGSTAB (Ours)']['converged'].append(info['converged'])
        
        bicgstab_str = f"{info['iterations']}" if info['converged'] else "FAIL"
        
        # Test BiCG
        x, info, elapsed = run_scipy_solver('bicg', A, b, x0, tol=1e-8, 
                                           maxiter=1000, M=M)
        err = np.linalg.norm(x - x_true)
        
        results['BiCG (SciPy)']['iterations'].append(info['iterations'])
        results['BiCG (SciPy)']['times'].append(elapsed)
        results['BiCG (SciPy)']['errors'].append(err)
        results['BiCG (SciPy)']['converged'].append(info['converged'])
        
        bicg_str = f"{info['iterations']}" if info['converged'] else "FAIL"
        
        # Test CGS
        x, info, elapsed = run_scipy_solver('cgs', A, b, x0, tol=1e-8, 
                                           maxiter=1000, M=M)
        err = np.linalg.norm(x - x_true)
        
        results['CGS (SciPy)']['iterations'].append(info['iterations'])
        results['CGS (SciPy)']['times'].append(elapsed)
        results['CGS (SciPy)']['errors'].append(err)
        results['CGS (SciPy)']['converged'].append(info['converged'])
        
        cgs_str = f"{info['iterations']}" if info['converged'] else "FAIL"
        
        # Test GMRES
        x, info, elapsed = run_scipy_solver('gmres', A, b, x0, tol=1e-8,
                                           maxiter=1000, restart=20, M=M)
        err = np.linalg.norm(x - x_true)
        
        # For GMRES, show inner iterations if available
        if 'cycles' in info:
            gmres_iters = info['iterations']  # Inner iterations
        else:
            gmres_iters = info['iterations'] * 20  # Estimate from cycles
        
        results['GMRES(20) (SciPy)']['iterations'].append(gmres_iters)
        results['GMRES(20) (SciPy)']['times'].append(elapsed)
        results['GMRES(20) (SciPy)']['errors'].append(err)
        results['GMRES(20) (SciPy)']['converged'].append(info['converged'])
        
        gmres_str = f"{gmres_iters}" if info['converged'] else "FAIL"
        
        # Print row
        print(f"{pe_actual:<8.1f} {vel_mag:<12.3f} {bicgstab_str:<12} {bicg_str:<12} "
              f"{cgs_str:<12} {gmres_str:<12}")
    
    print("="*80)
    
    return results


def plot_peclet_sweep(results, use_precond=False):
    """
    Visualize solver performance vs Péclet number
    """
    peclets = results['peclet_numbers']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {
        'BiCGSTAB (Ours)': 'blue',
        'BiCG (SciPy)': 'red',
        'CGS (SciPy)': 'green',
        'GMRES(20) (SciPy)': 'purple'
    }
    
    markers = {
        'BiCGSTAB (Ours)': 'o',
        'BiCG (SciPy)': 's',
        'CGS (SciPy)': '^',
        'GMRES(20) (SciPy)': 'D'
    }
    
    # ========================================================================
    # Plot 1: Iterations vs Péclet (log-log)
    # ========================================================================
    ax = axes[0, 0]
    
    for solver in ['BiCGSTAB (Ours)', 'BiCG (SciPy)', 'CGS (SciPy)', 'GMRES(20) (SciPy)']:
        iters = results[solver]['iterations']
        converged = results[solver]['converged']
        
        # Only plot converged points
        pe_conv = [pe for pe, conv in zip(peclets, converged) if conv]
        it_conv = [it for it, conv in zip(iters, converged) if conv]
        
        ax.loglog(pe_conv, it_conv, marker=markers[solver], 
                 color=colors[solver], linewidth=2.5, markersize=8,
                 label=solver)
    
    ax.set_xlabel('Péclet Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Iterations to Convergence', fontsize=13, fontweight='bold')
    ax.set_title('Iteration Count vs Péclet Number', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add regime labels
    ax.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(0.2, ax.get_ylim()[1]*0.8, 'Diffusion\nDominated', 
           fontsize=9, ha='center', alpha=0.7)
    ax.text(10, ax.get_ylim()[1]*0.8, 'Convection\nDominated', 
           fontsize=9, ha='center', alpha=0.7)
    
    # ========================================================================
    # Plot 2: Solve Time vs Péclet
    # ========================================================================
    ax = axes[0, 1]
    
    for solver in ['BiCGSTAB (Ours)', 'BiCG (SciPy)', 'CGS (SciPy)', 'GMRES(20) (SciPy)']:
        times = results[solver]['times']
        converged = results[solver]['converged']
        
        pe_conv = [pe for pe, conv in zip(peclets, converged) if conv]
        t_conv = [t for t, conv in zip(times, converged) if conv]
        
        ax.loglog(pe_conv, t_conv, marker=markers[solver], 
                 color=colors[solver], linewidth=2.5, markersize=8,
                 label=solver)
    
    ax.set_xlabel('Péclet Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Computational Time vs Péclet Number', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # ========================================================================
    # Plot 3: Relative Performance (BiCGSTAB = baseline)
    # ========================================================================
    ax = axes[1, 0]
    
    bicgstab_iters = results['BiCGSTAB (Ours)']['iterations']
    
    for solver in ['BiCG (SciPy)', 'CGS (SciPy)', 'GMRES(20) (SciPy)']:
        iters = results[solver]['iterations']
        converged_bicg = results['BiCGSTAB (Ours)']['converged']
        converged_solver = results[solver]['converged']
        
        # Only compare when both converged
        pe_both = [pe for pe, cb, cs in zip(peclets, converged_bicg, converged_solver) 
                   if cb and cs]
        relative = [it_s / it_b for it_s, it_b, cb, cs in 
                   zip(iters, bicgstab_iters, converged_bicg, converged_solver)
                   if cb and cs]
        
        ax.semilogx(pe_both, relative, marker=markers[solver],
                   color=colors[solver], linewidth=2.5, markersize=8,
                   label=f'{solver} / BiCGSTAB')
    
    ax.axhline(1.0, color='blue', linestyle='--', linewidth=2, 
              label='BiCGSTAB (baseline)', alpha=0.7)
    ax.set_xlabel('Péclet Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Iteration Ratio (relative to BiCGSTAB)', fontsize=13, fontweight='bold')
    ax.set_title('Relative Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # ========================================================================
    # Plot 4: Success Rate
    # ========================================================================
    ax = axes[1, 1]
    
    n_peclets = len(peclets)
    x_pos = np.arange(len(['BiCGSTAB', 'BiCG', 'CGS', 'GMRES']))
    
    success_rates = []
    for solver in ['BiCGSTAB (Ours)', 'BiCG (SciPy)', 'CGS (SciPy)', 'GMRES(20) (SciPy)']:
        n_converged = sum(results[solver]['converged'])
        success_rate = 100 * n_converged / n_peclets
        success_rates.append(success_rate)
    
    bars = ax.bar(x_pos, success_rates, 
                  color=['blue', 'red', 'green', 'purple'], alpha=0.7)
    ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Convergence Success Rate', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['BiCGSTAB', 'BiCG', 'CGS', 'GMRES'], fontsize=11)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{rate:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Overall title
    precond_str = "With ILU(0)" if use_precond else "No Preconditioning"
    plt.suptitle(f'Solver Performance vs Péclet Number ({precond_str})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def analyze_peclet_regimes(results):
    """
    Analyze solver behavior in different flow regimes
    """
    peclets = results['peclet_numbers']
    
    print("\n" + "="*80)
    print("PÉCLET REGIME ANALYSIS")
    print("="*80)
    
    # Define regimes
    regimes = {
        'Diffusion-dominated (Pe < 1)': lambda pe: pe < 1,
        'Balanced (1 ≤ Pe < 10)': lambda pe: 1 <= pe < 10,
        'Moderate convection (10 ≤ Pe < 100)': lambda pe: 10 <= pe < 100,
        'Strong convection (Pe ≥ 100)': lambda pe: pe >= 100
    }
    
    for regime_name, regime_filter in regimes.items():
        print(f"\n{regime_name}:")
        print("-"*70)
        
        # Find Péclet numbers in this regime
        pe_in_regime = [pe for pe in peclets if regime_filter(pe)]
        
        if not pe_in_regime:
            print("  No test points in this regime")
            continue
        
        indices = [i for i, pe in enumerate(peclets) if regime_filter(pe)]
        
        print(f"  Péclet numbers tested: {pe_in_regime}")
        print(f"\n  {'Solver':<25} {'Avg Iters':<12} {'Success Rate':<15}")
        print("  " + "-"*52)
        
        for solver in ['BiCGSTAB (Ours)', 'BiCG (SciPy)', 'CGS (SciPy)', 'GMRES(20) (SciPy)']:
            iters_regime = [results[solver]['iterations'][i] for i in indices]
            conv_regime = [results[solver]['converged'][i] for i in indices]
            
            # Average iterations (only for converged)
            iters_conv = [it for it, conv in zip(iters_regime, conv_regime) if conv]
            avg_iters = np.mean(iters_conv) if iters_conv else float('inf')
            
            # Success rate
            n_success = sum(conv_regime)
            success_rate = 100 * n_success / len(conv_regime)
            
            avg_str = f"{avg_iters:.1f}" if avg_iters != float('inf') else "N/A"
            print(f"  {solver:<25} {avg_str:<12} {success_rate:<14.0f}%")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)
    
    # Find BiCGSTAB performance trend
    bicgstab_iters = results['BiCGSTAB (Ours)']['iterations']
    low_pe_avg = np.mean([it for pe, it in zip(peclets, bicgstab_iters) if pe < 10])
    high_pe_avg = np.mean([it for pe, it in zip(peclets, bicgstab_iters) if pe >= 100])


def compare_with_without_precond(nx=40, ny=40, diffusion=0.01, 
                                peclet_targets=None):
    """
    Compare performance with and without preconditioning across Péclet range
    """
    print("\n" + "█"*80)
    print("PRECONDITIONING IMPACT ACROSS PÉCLET NUMBERS")
    print("█"*80)
    
    # Without preconditioning
    print("\n--- WITHOUT PRECONDITIONING ---")
    results_no_precond = sweep_peclet_numbers(nx, ny, diffusion, peclet_targets,
                                             use_precond=False)
    
    # With preconditioning
    print("\n--- WITH ILU(0) PRECONDITIONING ---")
    results_with_precond = sweep_peclet_numbers(nx, ny, diffusion, peclet_targets,
                                               use_precond=True)
    
    # Plot comparison
    plot_precond_comparison(results_no_precond, results_with_precond)
    
    return results_no_precond, results_with_precond


def plot_precond_comparison(results_no, results_with):
    """
    Side-by-side comparison with/without preconditioning
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    peclets_no = results_no['peclet_numbers']
    peclets_with = results_with['peclet_numbers']
    
    colors = {'BiCGSTAB (Ours)': 'blue', 'BiCG (SciPy)': 'red', 
              'CGS (SciPy)': 'green', 'GMRES(20) (SciPy)': 'purple'}
    markers = {'BiCGSTAB (Ours)': 'o', 'BiCG (SciPy)': 's',
               'CGS (SciPy)': '^', 'GMRES(20) (SciPy)': 'D'}
    
    # Plot 1: No preconditioning
    ax = axes[0]
    for solver in ['BiCGSTAB (Ours)', 'BiCG (SciPy)', 'CGS (SciPy)', 'GMRES(20) (SciPy)']:
        iters = results_no[solver]['iterations']
        converged = results_no[solver]['converged']
        
        pe_conv = [pe for pe, conv in zip(peclets_no, converged) if conv]
        it_conv = [it for it, conv in zip(iters, converged) if conv]
        
        ax.loglog(pe_conv, it_conv, marker=markers[solver],
                 color=colors[solver], linewidth=2.5, markersize=8,
                 label=solver)
    
    ax.set_xlabel('Péclet Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Iterations', fontsize=13, fontweight='bold')
    ax.set_title('No Preconditioning', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: With preconditioning
    ax = axes[1]
    for solver in ['BiCGSTAB (Ours)', 'BiCG (SciPy)', 'CGS (SciPy)', 'GMRES(20) (SciPy)']:
        iters = results_with[solver]['iterations']
        converged = results_with[solver]['converged']
        
        pe_conv = [pe for pe, conv in zip(peclets_with, converged) if conv]
        it_conv = [it for it, conv in zip(iters, converged) if conv]
        
        ax.loglog(pe_conv, it_conv, marker=markers[solver],
                 color=colors[solver], linewidth=2.5, markersize=8,
                 label=solver)
    
    ax.set_xlabel('Péclet Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Iterations', fontsize=13, fontweight='bold')
    ax.set_title('With ILU(0) Preconditioning', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Preconditioning Impact Across Péclet Range',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()