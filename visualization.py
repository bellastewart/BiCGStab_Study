"""
Visualization Functions for Solver Comparison

Authors: Viki Mancoridis & Bella Stewart
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_convergence_comparison(results_dict, problem_name, save_path=None):
    """
    Plot residual histories for all solvers
    """
    plt.figure(figsize=(12, 7))

    colors = {
        'BiCGSTAB (Ours)': 'blue',
        'BiCG (SciPy)': 'red',
        'CGS (SciPy)': 'green',
        'GMRES(20) (SciPy)': 'purple',
        'CG (SciPy)': 'orange'
    }

    markers = {
        'BiCGSTAB (Ours)': 'o',
        'BiCG (SciPy)': 's',
        'CGS (SciPy)': '^',
        'GMRES(20) (SciPy)': 'D',
        'CG (SciPy)': 'v'
    }

    for solver_name, result in results_dict.items():
        if 'residuals' not in result['info']:
            continue

        residuals = result['info']['residuals']
        if len(residuals) == 0:
            continue

        iters = np.arange(len(residuals))
        
        # Create label with appropriate iteration info
        if 'cycles' in result['info']:
            # GMRES: Show inner iterations in legend
            inner_iters = result['info']['iterations']
            cycles = result['info']['cycles']
            label = f"{solver_name} (~{inner_iters} inner iters, {cycles} cycles)"
        else:
            # Other solvers: Show regular iterations
            label = f"{solver_name} ({result['info']['iterations']} iters)"
        
        plt.semilogy(iters, residuals,
                    label=label,
                    color=colors.get(solver_name, 'black'),
                    marker=markers.get(solver_name, 'o'),
                    markevery=max(1, len(residuals)//10),
                    linewidth=2.5, markersize=7)

    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Residual Norm ||r_k||', fontsize=14, fontweight='bold')
    plt.title(f'Convergence Comparison: {problem_name}',
             fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cgs_spiking_demo(results_dict, problem_name):
    """
    Demonstrate CGS spiking vs BiCGSTAB smoothness
    """
    # Extract residuals
    cgs_residuals = results_dict['CGS (SciPy)']['info']['residuals']
    bicgstab_residuals = results_dict['BiCGSTAB (Ours)']['info']['residuals']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Linear scale - shows spikes dramatically
    ax1.plot(cgs_residuals, 'g-', linewidth=2.5, label='CGS (SciPy)',
            marker='^', markevery=max(1, len(cgs_residuals)//15))
    ax1.plot(bicgstab_residuals, 'b-', linewidth=2.5, label='BiCGSTAB (Ours)',
            marker='o', markevery=max(1, len(bicgstab_residuals)//15))
    ax1.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax1.set_ylabel('||r_k|| (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.set_title('CGS Residual Spikes vs BiCGSTAB Smoothness',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.4)

    # Log scale - shows overall convergence
    ax2.semilogy(cgs_residuals, 'g-', linewidth=2.5, label='CGS (SciPy)',
                marker='^', markevery=max(1, len(cgs_residuals)//15))
    ax2.semilogy(bicgstab_residuals, 'b-', linewidth=2.5, label='BiCGSTAB (Ours)',
                marker='o', markevery=max(1, len(bicgstab_residuals)//15))
    ax2.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax2.set_ylabel('||r_k|| (Log Scale)', fontsize=13, fontweight='bold')
    ax2.set_title('Convergence Comparison (Log Scale)',
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.4)

    plt.suptitle(f'{problem_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # Compute and display spike ratios
    cgs_max_ratio = max(cgs_residuals[i]/cgs_residuals[i-1]
                       for i in range(1, len(cgs_residuals))
                       if cgs_residuals[i-1] > 0)
    bicgstab_max_ratio = max(bicgstab_residuals[i]/bicgstab_residuals[i-1]
                            for i in range(1, len(bicgstab_residuals))
                            if bicgstab_residuals[i-1] > 0)

    print(f"\nMax residual increase ratio:")
    print(f"  CGS (SciPy):     {cgs_max_ratio:,.2f}×")
    print(f"  BiCGSTAB (Ours): {bicgstab_max_ratio:.2f}×")
    print(f"  → BiCGSTAB is {cgs_max_ratio/bicgstab_max_ratio:.1f}× more stable!")


def create_summary_table(all_results):
    """
    Create summary table of all test results
    """
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Problem':<25} {'Solver':<20} {'Iters':<8} {'Error':<12} {'Time(s)':<10}")
    print("-"*80)

    for prob_name, results in all_results.items():
        for i, (solver, res) in enumerate(results.items()):
            prob_str = prob_name if i == 0 else ""
            iters = res['info']['iterations']
            error = f"{res['error']:.2e}"
            time_s = f"{res['time']:.3f}"

            print(f"{prob_str:<25} {solver:<20} {iters:<8} {error:<12} {time_s:<10}")
        print("-"*80)
    print("="*80)