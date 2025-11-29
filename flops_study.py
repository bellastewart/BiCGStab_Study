"""
Detailed FLOPs Analysis and Comparison

Authors: Viki Mancoridis & Bella Stewart
"""

import numpy as np
import matplotlib.pyplot as plt
from helpers import estimate_flops, compare_computational_work, analyze_work_breakdown


def detailed_flop_analysis(results_dict, A, problem_name):
    """
    Comprehensive FLOP analysis with visualizations
    """
    print("\n" + "█"*80)
    print(f"DETAILED FLOP ANALYSIS: {problem_name}")
    print("█"*80)
    
    # Compute work comparison
    work_comparison = compare_computational_work(results_dict, A)
    
    # Analyze breakdown
    analyze_work_breakdown(work_comparison)
    
    # Plot visualizations
    plot_flop_comparison(work_comparison, results_dict, A, problem_name)
    
    return work_comparison


def plot_flop_comparison(work_comparison, results_dict, A, problem_name):
    """
    Visualize FLOP comparison across solvers
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    solvers = list(work_comparison.keys())
    colors = {
        'BiCGSTAB (Ours)': 'blue',
        'BiCG (SciPy)': 'red',
        'CGS (SciPy)': 'green',
        'GMRES(20) (SciPy)': 'purple',
        'CG (SciPy)': 'orange'
    }
    
    # ========================================================================
    # Plot 1: Total FLOPs Comparison
    # ========================================================================
    ax = axes[0, 0]
    
    total_flops = [work_comparison[s]['total_flops'] for s in solvers]
    x_pos = np.arange(len(solvers))
    
    bars = ax.bar(x_pos, total_flops, 
                color=[colors.get(s, 'gray') for s in solvers],
                alpha=0.7)
    
    ax.set_ylabel('Total FLOPs', fontsize=13, fontweight='bold')
    ax.set_title('Total Computational Work', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace(' (Ours)', '').replace(' (SciPy)', '') 
                        for s in solvers], rotation=45, ha='right', fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, flops in zip(bars, total_flops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
            f'{flops/1e6:.1f}M', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
    
    # ========================================================================
    # Plot 2: FLOPs per Iteration
    # ========================================================================
    ax = axes[0, 1]
    
    flops_per_iter = [work_comparison[s]['flops_per_iter'] for s in solvers]
    
    bars = ax.bar(x_pos, flops_per_iter,
                color=[colors.get(s, 'gray') for s in solvers],
                alpha=0.7)
    
    ax.set_ylabel('FLOPs per Iteration', fontsize=13, fontweight='bold')
    ax.set_title('Cost per Iteration', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace(' (Ours)', '').replace(' (SciPy)', '') 
                        for s in solvers], rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, fpi in zip(bars, flops_per_iter):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
            f'{fpi/1e3:.0f}K', ha='center', va='bottom',
            fontsize=9, fontweight='bold')


def analyze_flop_efficiency(work_comparison, results_dict):
    """
    Analyze computational efficiency metrics
    """
    print("\n" + "="*80)
    print("COMPUTATIONAL EFFICIENCY METRICS")
    print("="*80)
    
    solvers = list(work_comparison.keys())
    
    # Find best performer (minimum FLOPs)
    min_flops = min(work_comparison[s]['total_flops'] for s in solvers)
    best_solver = min(solvers, key=lambda s: work_comparison[s]['total_flops'])
    
    print(f"\n{'Solver':<25} {'FLOPs/Iter':<15} {'Iterations':<12} "
          f"{'Total FLOPs':<18} {'Efficiency':<12}")
    print("-"*90)
    
    for solver in solvers:
        flops_per_iter = work_comparison[solver]['flops_per_iter']
        iterations = results_dict[solver]['info']['iterations']
        total_flops = work_comparison[solver]['total_flops']
        efficiency = min_flops / total_flops * 100
        
        print(f"{solver:<25} {flops_per_iter:<15,.0f} {iterations:<12} "
              f"{total_flops:<18,} {efficiency:<12.1f}%")
    
    print("-"*90)
    print(f"Most efficient: {best_solver}")
    
    # Analyze SpMV efficiency
    print("\n" + "="*80)
    print("SpMV EFFICIENCY ANALYSIS")
    print("="*80)
    print(f"\n{'Solver':<25} {'SpMV Count':<12} {'SpMV/Iter':<12} "
          f"{'SpMV % of Work':<15}")
    print("-"*70)
    
    for solver in solvers:
        if work_comparison[solver]['breakdown']:
            spmv_count = work_comparison[solver]['spmv_count']
            iterations = results_dict[solver]['info']['iterations']
            spmv_per_iter = spmv_count / iterations if iterations > 0 else 0
            
            spmv_flops = work_comparison[solver]['breakdown']['spmv']
            total_flops = work_comparison[solver]['total_flops']
            spmv_pct = spmv_flops / total_flops * 100
            
            print(f"{solver:<25} {spmv_count:<12} {spmv_per_iter:<12.1f} "
                  f"{spmv_pct:<15.1f}%")


def compare_memory_vs_flops(work_comparison, results_dict, A):
    """
    Compare memory usage vs computational cost tradeoff
    """
    print("\n" + "="*80)
    print("MEMORY vs FLOPs TRADEOFF")
    print("="*80)
    
    n = A.shape[0]
    vector_size_bytes = n * 8  # float64
    
    # Memory requirements (approximate)
    memory_usage = {
        'BiCGSTAB (Ours)': 6 * vector_size_bytes,  # x, r, r_hat, p, v, s, t
        'BiCG (SciPy)': 6 * vector_size_bytes,     # Similar to BiCGSTAB
        'CGS (SciPy)': 6 * vector_size_bytes,      # Similar
        'GMRES(20) (SciPy)': 20 * vector_size_bytes,  # Krylov basis
        'CG (SciPy)': 4 * vector_size_bytes        # x, r, p, Ap
    }
    
    print(f"\n{'Solver':<25} {'Memory (KB)':<15} {'Total FLOPs':<18} "
          f"{'FLOPs/Memory':<15}")
    print("-"*80)
    
    for solver in work_comparison.keys():
        if solver in memory_usage:
            mem_kb = memory_usage[solver] / 1024
            total_flops = work_comparison[solver]['total_flops']
            flops_per_kb = total_flops / mem_kb
            
            print(f"{solver:<25} {mem_kb:<15,.1f} {total_flops:<18,} "
                  f"{flops_per_kb:<15,.0f}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 7))
    
    solvers = list(work_comparison.keys())
    colors = {
        'BiCGSTAB (Ours)': 'blue',
        'BiCG (SciPy)': 'red',
        'CGS (SciPy)': 'green',
        'GMRES(20) (SciPy)': 'purple',
        'CG (SciPy)': 'orange'
    }
    
    for solver in solvers:
        if solver in memory_usage:
            mem_kb = memory_usage[solver] / 1024
            total_flops = work_comparison[solver]['total_flops']
            
            # Size proportional to iterations
            iters = results_dict[solver]['info']['iterations']
            size = 50 + iters * 2
            
            ax.scatter(mem_kb, total_flops, s=size,
                      color=colors.get(solver, 'gray'), alpha=0.7,
                      edgecolors='black', linewidth=2,
                      label=f"{solver.replace(' (Ours)', '').replace(' (SciPy)', '')} ({iters} it)")
    
    ax.set_xlabel('Memory Usage (KB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total FLOPs', fontsize=13, fontweight='bold')
    ax.set_title('Memory vs Computational Cost Tradeoff',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    # Pareto frontier annotation
    ax.text(0.02, 0.98, 'Lower-left = Better\n(Less memory, fewer FLOPs)',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()