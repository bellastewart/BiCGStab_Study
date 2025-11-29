"""
Preconditioner Construction

Also, scipy wrappers for fair comparison

Authors: Viki Mancoridis & Bella Stewart
"""

from scipy.sparse.linalg import spilu, LinearOperator
import numpy as np
import scipy.sparse.linalg as spla
import time


def build_ilu_preconditioner(A, fill_factor=1.0, drop_tol=1e-4):
    """
    Build ILU preconditioner with variable strength
    """
    try:
        ilu = spilu(A.tocsc(), fill_factor=fill_factor, drop_tol=drop_tol)
        
        # Count nonzeros in ILU factors
        nnz = ilu.L.nnz + ilu.U.nnz
        
        # Create LinearOperator with both matvec and rmatvec
        # matvec: applies M^{-1} @ v
        # rmatvec: applies (M^{-1})^T @ v = (M^T)^{-1} @ v
        M_inv = LinearOperator(
            A.shape, 
            matvec=ilu.solve,
            rmatvec=lambda v: ilu.solve(v, trans='T')  # Transpose solve
        )
        
        return M_inv, True, nnz
    except Exception as e:
        print(f"ILU factorization failed: {e}")
        return None, False, 0



def run_scipy_solver(solver_name, A, b, x0=None, tol=1e-8, maxiter=1000, restart=20, M=None):
    """
    Wrapper for SciPy solvers with residual tracking
    """
    if x0 is None:
        x0 = np.zeros_like(b)

    residuals = []
    r0 = b - A.dot(x0)
    residuals.append(np.linalg.norm(r0))

    if solver_name == 'gmres':
        # For GMRES: Track restart cycles AND compute inner iterations
        cycle_count = [0]
        
        def gmres_callback_x(xk):
            """Called once per restart cycle"""
            cycle_count[0] += 1
            rk = b - A.dot(xk)
            residuals.append(np.linalg.norm(rk))
        
        t0 = time.time()
        x, exit_code = spla.gmres(A, b, x0=x0, rtol=tol, atol=0,
                                  maxiter=maxiter, restart=restart, M=M,
                                  callback=gmres_callback_x,
                                  callback_type='x')  # ← Back to 'x' for cycle-based
        elapsed = time.time() - t0
        
        # Cycles = number of callbacks
        cycles = cycle_count[0]
        
        # Estimate inner iterations
        # GMRES does up to 'restart' iterations per cycle
        # Final cycle may be partial
        if exit_code == 0:  # Converged
            # Estimate: (cycles - 1) * restart + partial_last_cycle
            # Conservative estimate: cycles * restart (upper bound)
            # Better estimate: Check if we hit restart limit
            inner_iters_estimate = cycles * restart
        else:
            inner_iters_estimate = cycles * restart
        
        info = {
            'converged': (exit_code == 0),
            'iterations': inner_iters_estimate,  # Inner iterations (estimated)
            'cycles': cycles,                     # Restart cycles (exact)
            'residuals': residuals,               # One per cycle for plotting
            'reason': 'converged' if exit_code == 0 else f'exit_code={exit_code}',
            'exit_code': exit_code,
            'restart': restart
        }
        
    else:
        # Standard callback for other solvers
        def callback(xk):
            rk = b - A.dot(xk)
            residuals.append(np.linalg.norm(rk))

        t0 = time.time()

        if solver_name == 'bicg':
            x, exit_code = spla.bicg(A, b, x0=x0, rtol=tol, atol=0,
                                     maxiter=maxiter, M=M, callback=callback)
        elif solver_name == 'cgs':
            x, exit_code = spla.cgs(A, b, x0=x0, rtol=tol, atol=0,
                                    maxiter=maxiter, M=M, callback=callback)
        elif solver_name == 'bicgstab':
            x, exit_code = spla.bicgstab(A, b, x0=x0, rtol=tol, atol=0,
                                         maxiter=maxiter, M=M, callback=callback)
        elif solver_name == 'cg':
            x, exit_code = spla.cg(A, b, x0=x0, rtol=tol, atol=0,
                                   maxiter=maxiter, M=M, callback=callback)
        else:
            raise ValueError(f"Unknown solver: {solver_name}")
        
        elapsed = time.time() - t0
        iterations = len(residuals) - 1
        
        info = {
            'converged': (exit_code == 0),
            'iterations': iterations,
            'residuals': residuals,
            'reason': 'converged' if exit_code == 0 else f'exit_code={exit_code}',
            'exit_code': exit_code,
            'restart': None
        }

    return x, info, elapsed


def estimate_flops(solver_name, A, iterations, restart=None):
    """
    Estimate FLOPs for solver based on iteration count
    """
    n = A.shape[0]
    nnz = A.nnz
    spmv_cost = 2 * nnz  # 2 FLOPs per nonzero (multiply + add)
    
    if solver_name in ['BiCGSTAB (Ours)', 'bicgstab']:
        # BiCGSTAB per iteration:
        # - 2 SpMV (v = A*z, t = A*y)
        # - ~10 vector operations (n FLOPs each):
        #   * rho = dot(r_hat, r)              : 2n
        #   * beta computation, p update       : 4n
        #   * r_hat_v = dot(r_hat, v)          : 2n
        #   * s = r - alpha*v                  : 2n
        #   * s_norm = norm(s)                 : 2n
        #   * t_dot_t = dot(t, t)              : 2n
        #   * omega = dot(t, s) / t_dot_t      : 2n
        #   * x update: x += alpha*z + omega*y : 3n
        #   * r update: r = s - omega*t        : 2n
        #   * r_norm = norm(r)                 : 2n
        #   Total: ~23n per iteration
        
        vector_ops_per_iter = 23 * n
        spmv_per_iter = 2
        
        total_spmv = spmv_per_iter * spmv_cost * iterations
        total_vector = vector_ops_per_iter * iterations
        total_flops = total_spmv + total_vector
        
        breakdown = {
            'spmv': total_spmv,
            'vector_ops': total_vector,
            'spmv_count': spmv_per_iter * iterations,
            'flops_per_iter': spmv_per_iter * spmv_cost + vector_ops_per_iter
        }
        
    elif solver_name in ['BiCG (SciPy)', 'bicg']:
        # BiCG per iteration:
        # - 2 SpMV (A*p, A^T*p_hat) - but we only count forward SpMV
        # - ~8 vector operations
        
        vector_ops_per_iter = 16 * n
        spmv_per_iter = 2  # A*p and A^T*p_hat (though A^T may not be formed)
        
        total_spmv = spmv_per_iter * spmv_cost * iterations
        total_vector = vector_ops_per_iter * iterations
        total_flops = total_spmv + total_vector
        
        breakdown = {
            'spmv': total_spmv,
            'vector_ops': total_vector,
            'spmv_count': spmv_per_iter * iterations,
            'flops_per_iter': spmv_per_iter * spmv_cost + vector_ops_per_iter
        }
        
    elif solver_name in ['CGS (SciPy)', 'cgs']:
        # CGS per iteration:
        # - 2 SpMV
        # - ~10 vector operations
        
        vector_ops_per_iter = 20 * n
        spmv_per_iter = 2
        
        total_spmv = spmv_per_iter * spmv_cost * iterations
        total_vector = vector_ops_per_iter * iterations
        total_flops = total_spmv + total_vector
        
        breakdown = {
            'spmv': total_spmv,
            'vector_ops': total_vector,
            'spmv_count': spmv_per_iter * iterations,
            'flops_per_iter': spmv_per_iter * spmv_cost + vector_ops_per_iter
        }
        
    elif solver_name in ['GMRES(20) (SciPy)', 'gmres']:
        # GMRES per inner iteration:
        # - 1 SpMV (A*v)
        # - Modified Gram-Schmidt: O(m*n) where m = current iteration in cycle
        #   Average m over restart cycle ≈ restart/2
        # - For simplicity, use average cost per iteration
        
        m = restart if restart else 20
        
        # Per inner iteration:
        # - 1 SpMV: spmv_cost
        # - Gram-Schmidt with j vectors (j = 1 to m, average m/2):
        #   * j dot products: 2*n*j each
        #   * j AXPY operations: 2*n*j
        #   Average over cycle: 2*n*(m/2) + 2*n*(m/2) = 2*n*m
        
        vector_ops_per_iter = 2 * n * m  # Gram-Schmidt
        spmv_per_iter = 1
        
        total_spmv = spmv_per_iter * spmv_cost * iterations
        total_vector = vector_ops_per_iter * iterations
        total_flops = total_spmv + total_vector
        
        breakdown = {
            'spmv': total_spmv,
            'vector_ops': total_vector,
            'spmv_count': spmv_per_iter * iterations,
            'flops_per_iter': spmv_per_iter * spmv_cost + vector_ops_per_iter,
            'restart': m
        }
        
    elif solver_name in ['CG (SciPy)', 'cg']:
        # CG per iteration:
        # - 1 SpMV (A*p)
        # - ~6 vector operations
        
        vector_ops_per_iter = 12 * n
        spmv_per_iter = 1
        
        total_spmv = spmv_per_iter * spmv_cost * iterations
        total_vector = vector_ops_per_iter * iterations
        total_flops = total_spmv + total_vector
        
        breakdown = {
            'spmv': total_spmv,
            'vector_ops': total_vector,
            'spmv_count': spmv_per_iter * iterations,
            'flops_per_iter': spmv_per_iter * spmv_cost + vector_ops_per_iter
        }
        
    else:
        return None, None
    
    return total_flops, breakdown


def compare_computational_work(results_dict, A):
    """
    Compare actual computational work (FLOPs) across solvers
    """
    print("\n" + "="*80)
    print("COMPUTATIONAL WORK COMPARISON (FLOPs)")
    print("="*80)
    print(f"Problem size: n = {A.shape[0]}, nnz = {A.nnz}")
    print(f"SpMV cost: {2*A.nnz:,} FLOPs")
    print("="*80)
    
    work_comparison = {}
    
    print(f"\n{'Solver':<25} {'Iterations':<12} {'Total FLOPs':<18} "
          f"{'FLOPs/Iter':<15} {'SpMV Count':<12}")
    print("-"*90)
    
    flops_dict = {}
    
    for solver, result in results_dict.items():
        iters = result['info']['iterations']
        
        # Get FLOPs
        if 'flops' in result['info'] and result['info']['flops'] is not None:
            # Our implementation tracks exactly
            total_flops = result['info']['flops']
            breakdown = None
        else:
            # Estimate for SciPy solvers
            restart = 20 if 'GMRES' in solver else None
            total_flops, breakdown = estimate_flops(solver, A, iters, restart)
        
        if total_flops is None:
            continue
        
        flops_dict[solver] = total_flops
        
        # FLOP per iteration
        flops_per_iter = total_flops / iters if iters > 0 else 0
        
        # SpMV count
        if breakdown:
            spmv_count = breakdown['spmv_count']
        else:
            # For our BiCGSTAB, estimate 2 per iteration
            spmv_count = 2 * iters
        
        work_comparison[solver] = {
            'total_flops': total_flops,
            'flops_per_iter': flops_per_iter,
            'spmv_count': spmv_count,
            'breakdown': breakdown
        }
        
        print(f"{solver:<25} {iters:<12} {total_flops:<18,} "
              f"{flops_per_iter:<15,.0f} {spmv_count:<12}")
    
    # Find minimum for relative comparison
    if flops_dict:
        min_flops = min(flops_dict.values())
        min_solver = min(flops_dict, key=flops_dict.get)
        
        print("\n" + "-"*90)
        print(f"{'Solver':<25} {'Total FLOPs':<18} {'Relative Cost':<15} "
              f"{'Efficiency':<15}")
        print("-"*90)
        
        for solver, flops in flops_dict.items():
            relative = flops / min_flops
            efficiency = min_flops / flops * 100
            
            print(f"{solver:<25} {flops:<18,} {relative:<15.2f}× "
                  f"{efficiency:<15.1f}%")
        
        print("-"*90)
        print(f"Most efficient: {min_solver} with {min_flops:,} FLOPs")
    
    print("="*90)
    
    return work_comparison


def analyze_work_breakdown(work_comparison):
    """
    Analyze breakdown of computational work by operation type
    
    Parameters:
    -----------
    work_comparison : dict
        Results from compare_computational_work
    """
    print("\n" + "="*80)
    print("WORK BREAKDOWN BY OPERATION TYPE")
    print("="*80)
    
    print(f"\n{'Solver':<25} {'SpMV FLOPs':<18} {'Vector FLOPs':<18} "
          f"{'SpMV %':<10}")
    print("-"*80)
    
    for solver, work in work_comparison.items():
        if work['breakdown']:
            spmv_flops = work['breakdown']['spmv']
            vector_flops = work['breakdown']['vector_ops']
            total = work['total_flops']
            
            spmv_pct = spmv_flops / total * 100 if total > 0 else 0
            
            print(f"{solver:<25} {spmv_flops:<18,} {vector_flops:<18,} "
                  f"{spmv_pct:<10.1f}%")