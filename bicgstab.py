"""
BiCGSTAB Implementation
"""

import numpy as np
from scipy.sparse.linalg import aslinearoperator


def bicgstab(A, b, x0=None, tol=1e-8, maxiter=None, M=None, callback=None):
    """
    BiCGSTAB (Bi-Conjugate Gradient Stabilized) Implementation

    This is our from-scratch implementation, validated against SciPy.
    Uses van der Vorst's 1992 algorithm with dual convergence checks.

    Parameters:
    -----------
    A : matrix or LinearOperator
        System matrix (can be nonsymmetric)
    b : array
        Right-hand side vector
    x0 : array, optional
        Initial guess (default: zeros)
    tol : float
        Relative tolerance for convergence
    maxiter : int, optional
        Maximum iterations (default: min(1000, n))
    M : LinearOperator, optional
        Preconditioner (applies M^{-1})
    callback : callable, optional
        Called as callback(iter, x, residual_norm)

    Returns:
    --------
    x : array
        Solution vector
    info : dict
        Convergence information with keys:
        - converged: bool
        - iterations: int
        - residuals: list of residual norms
        - reason: str
        - flops: int (approximate)
    """
    A = aslinearoperator(A)
    n = b.shape[0]

    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = x0.copy().astype(float)

    if maxiter is None:
        maxiter = min(1000, n)

    # Preconditioner
    if M is None:
        def apply_M(v): return v
    else:
        M = aslinearoperator(M)
        def apply_M(v): return M.matvec(v)

    # Initialize
    r = b - A.matvec(x)
    r_hat = r.copy()  # Shadow residual
    rho_old = alpha = omega = 1.0
    v = p = np.zeros_like(b, dtype=float)

    res0 = np.linalg.norm(r)
    if res0 == 0:
        return x, {'converged': True, 'iterations': 0, 'residuals': [0.0],
                   'reason': 'zero rhs', 'flops': 0}

    tol_abs = tol * res0
    residuals = [res0]
    converged = False
    reason = ''
    flops = 2*n + 2*n  # initial SpMV + norm

    # Main iteration loop
    for iter_count in range(1, maxiter+1):
        # Bi-Lanczos step
        rho = np.dot(r_hat, r)
        flops += 2*n

        if abs(rho) < 1e-16:
            reason = 'rho breakdown'
            break

        if iter_count == 1:
            p = r.copy()
        else:
            beta = (rho/rho_old) * (alpha/omega)
            p = r + beta * (p - omega * v)
            flops += 4*n

        z = apply_M(p)
        v = A.matvec(z)
        flops += 2*n  # SpMV #1

        r_hat_v = np.dot(r_hat, v)
        flops += 2*n

        if abs(r_hat_v) < 1e-16:
            reason = 'alpha breakdown'
            break

        alpha = rho / r_hat_v
        s = r - alpha * v
        flops += 2*n

        # Check for early convergence
        s_norm = np.linalg.norm(s)
        flops += 2*n

        if s_norm <= tol_abs:
            x += alpha * z
            residuals.append(s_norm)
            converged = True
            reason = 'converged_s'
            if callback: callback(iter_count, x, s_norm)
            break

        # Stabilization step
        y = apply_M(s)
        t = A.matvec(y)
        flops += 2*n  # SpMV #2

        t_dot_t = np.dot(t, t)
        flops += 2*n

        if t_dot_t == 0:
            reason = 'omega breakdown (t zero)'
            break

        omega = np.dot(t, s) / t_dot_t
        flops += 2*n

        if abs(omega) < 1e-16:
            reason = 'omega breakdown (small)'
            break

        # Update solution and residual
        x += alpha * z + omega * y
        r = s - omega * t
        flops += 6*n

        r_norm = np.linalg.norm(r)
        flops += 2*n
        residuals.append(r_norm)

        if callback: callback(iter_count, x, r_norm)

        if r_norm <= tol_abs:
            converged = True
            reason = 'converged_r'
            break

        if not np.isfinite(r_norm):
            reason = 'non-finite residual'
            break

        rho_old = rho
    else:
        iter_count = maxiter
        reason = 'maxiter'

    info = {'converged': converged, 'iterations': iter_count,
            'residuals': residuals, 'reason': reason, 'flops': flops}
    return x, info