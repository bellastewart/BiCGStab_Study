"""
Preconditioner Construction

"""

from scipy.sparse.linalg import spilu, LinearOperator


def build_ilu_preconditioner(A, fill_factor=1.0, drop_tol=1e-4):
    """
    Build ILU(0) preconditioner using scipy.sparse.linalg.spilu

    Parameters:
    -----------
    A : sparse matrix
        System matrix
    fill_factor : float
        Amount of fill-in allowed (1.0 = ILU(0))
    drop_tol : float
        Drop tolerance for small entries

    Returns:
    --------
    M_inv : LinearOperator
        Preconditioner that applies M^{-1}
    success : bool
        Whether factorization succeeded
    """
    try:
        ilu = spilu(A.tocsc(), fill_factor=fill_factor, drop_tol=drop_tol)
        M_inv = LinearOperator(A.shape, matvec=ilu.solve)
        return M_inv, True
    except Exception as e:
        print(f"ILU factorization failed: {e}")
        return None, False


"""
SciPy Solver Wrappers for Fair Comparison

"""

import numpy as np
import scipy.sparse.linalg as spla
import time


def run_scipy_solver(solver_name, A, b, x0=None, tol=1e-8, maxiter=1000, restart=20):
    """
    Wrapper for SciPy solvers with residual tracking

    Parameters:
    -----------
    solver_name : str
        One of: 'bicg', 'cgs', 'bicgstab', 'gmres', 'cg'
    A, b, x0 : as in bicgstab()
    tol : relative tolerance
    maxiter : maximum iterations
    restart : GMRES restart parameter

    Returns:
    --------
    x : solution vector
    info : dict with same structure as bicgstab()
    elapsed : wall-clock time
    """
    if x0 is None:
        x0 = np.zeros_like(b)

    residuals = []
    r0 = b - A.dot(x0)
    residuals.append(np.linalg.norm(r0))

    def callback(xk):
        rk = b - A.dot(xk)
        residuals.append(np.linalg.norm(rk))

    t0 = time.time()

    if solver_name == 'bicg':
        x, exit_code = spla.bicg(A, b, x0=x0, rtol=tol, atol=0,
                                 maxiter=maxiter, callback=callback)
    elif solver_name == 'cgs':
        x, exit_code = spla.cgs(A, b, x0=x0, rtol=tol, atol=0,
                                maxiter=maxiter, callback=callback)
    elif solver_name == 'bicgstab':
        x, exit_code = spla.bicgstab(A, b, x0=x0, rtol=tol, atol=0,
                                     maxiter=maxiter, callback=callback)
    elif solver_name == 'gmres':
        x, exit_code = spla.gmres(A, b, x0=x0, rtol=tol, atol=0,
                                  maxiter=maxiter, restart=restart,
                                  callback=callback, callback_type='x')
    elif solver_name == 'cg':
        x, exit_code = spla.cg(A, b, x0=x0, rtol=tol, atol=0,
                               maxiter=maxiter, callback=callback)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    elapsed = time.time() - t0

    info = {
        'converged': (exit_code == 0),
        'iterations': len(residuals) - 1,
        'residuals': residuals,
        'reason': 'converged' if exit_code == 0 else f'exit_code={exit_code}',
        'exit_code': exit_code
    }

    return x, info, elapsed