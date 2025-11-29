"""
Test problem generators for sparse linear systems

Authors: Viki Mancoridis & Bella Stewart
"""

import numpy as np
import scipy.sparse as sp

def poisson_2d(nx, ny):
    """
    2D Poisson equation -∇²u = f on unit square with Dirichlet BC
    
    Returns symmetric positive definite matrix.
    """
    N = nx * ny
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)
    
    def idx(i, j):
        return i + j * nx
    
    rows, cols, data = [], [], []
    
    for j in range(ny):
        for i in range(nx):
            row = idx(i, j)
            
            data.append(4.0 / (dx * dy))
            rows.append(row)
            cols.append(row)
            
            if i > 0:
                rows.append(row)
                cols.append(idx(i-1, j))
                data.append(-1.0 / (dx * dy))
            if i < nx-1:
                rows.append(row)
                cols.append(idx(i+1, j))
                data.append(-1.0 / (dx * dy))
            if j > 0:
                rows.append(row)
                cols.append(idx(i, j-1))
                data.append(-1.0 / (dx * dy))
            if j < ny-1:
                rows.append(row)
                cols.append(idx(i, j+1))
                data.append(-1.0 / (dx * dy))
    
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A


def convection_diffusion_2d(nx, ny, diffusion=1.0, vel=(1.0, 0.0)):
    """
    2D convection-diffusion equation on unit square
    
    -ε∇²u + v·∇u = f with upwind discretization
    
    Returns nonsymmetric matrix.
    """
    N = nx * ny
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)
    eps = diffusion
    vx, vy = vel
    
    def idx(i, j):
        return i + j * nx
    
    rows, cols, data = [], [], []
    
    for j in range(ny):
        for i in range(nx):
            row = idx(i, j)
            center = 0.0
            
            center += 2*eps*(1/dx**2 + 1/dy**2)
            
            if i > 0:
                rows.append(row)
                cols.append(idx(i-1, j))
                data.append(-eps/dx**2)
            if i < nx-1:
                rows.append(row)
                cols.append(idx(i+1, j))
                data.append(-eps/dx**2)
            if j > 0:
                rows.append(row)
                cols.append(idx(i, j-1))
                data.append(-eps/dy**2)
            if j < ny-1:
                rows.append(row)
                cols.append(idx(i, j+1))
                data.append(-eps/dy**2)
            
            # Convection (upwind)
            if vx > 0:
                if i > 0:
                    rows.append(row)
                    cols.append(idx(i-1, j))
                    data.append(-vx/dx)
                center += vx/dx
            elif vx < 0:
                if i < nx-1:
                    rows.append(row)
                    cols.append(idx(i+1, j))
                    data.append(vx/dx)
                center -= vx/dx
            
            if vy > 0:
                if j > 0:
                    rows.append(row)
                    cols.append(idx(i, j-1))
                    data.append(-vy/dy)
                center += vy/dy
            elif vy < 0:
                if j < ny-1:
                    rows.append(row)
                    cols.append(idx(i, j+1))
                    data.append(vy/dy)
                center -= vy/dy
            
            rows.append(row)
            cols.append(row)
            data.append(center)
    
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A