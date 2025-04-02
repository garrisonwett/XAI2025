import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EPS = np.finfo(float).eps

# Triangular membership function.
def triangular_mf(x, a, b, c):
    x_arr = np.array(x, dtype=float)
    mu = np.zeros_like(x_arr)
    left_mask  = (x_arr > a) & (x_arr < b)
    right_mask = (x_arr >= b) & (x_arr < c)
    mu[left_mask]  = (x_arr[left_mask] - a) / (b - a)
    mu[right_mask] = (c - x_arr[right_mask]) / (c - b)
    return mu.item() if np.isscalar(x) else mu

# Build triangular membership functions given centers.
def build_triangles(centers):
    sorted_centers = np.sort(centers)
    # Include 0 and 1 as endpoints.
    full = np.concatenate(([0.0], sorted_centers, [1.0]))
    mfs = []
    n = len(full)
    for i in range(n):
        left   = full[i-1] if i > 0 else full[0]
        center = full[i]
        right  = full[i+1] if i < n-1 else full[-1]
        # Capture current values using default arguments.
        mfs.append(lambda x, l=left, c=center, r=right: triangular_mf(x, l, c, r))
    return mfs

# 2D TSK inference (constant output) using two inputs and two sets of MFs.
def tsk_inference_const_2d(x1, x2, mfs1, mfs2, rule_constants):
    num = 0.0
    den = 0.0
    for i, mf1 in enumerate(mfs1):
        for j, mf2 in enumerate(mfs2):
            w = mf1(x1) * mf2(x2)
            y = rule_constants[i, j]
            num += w * y
            den += w
    return num / (den + EPS)

# Compute a FIS surface over a grid of two inputs.
def compute_fis_surface(mfs1, mfs2, rule_constants, resolution=50):
    x1_vals = np.linspace(EPS, 1 - EPS, resolution)
    x2_vals = np.linspace(EPS, 1 - EPS, resolution)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Y = np.zeros_like(X1)
    
    # Evaluate the FIS output at every grid point.
    for i in range(resolution):
        for j in range(resolution):
            Y[i, j] = tsk_inference_const_2d(X1[i, j], X2[i, j], mfs1, mfs2, rule_constants)
    
    return X1, X2, Y

# Main visualization routine for two 2D FIS surfaces.
def visualize_2d_fis_surfaces():
    # For demonstration, use one center for each input.
    # With centers = [0.5] the membership functions are defined at 0, 0.5, and 1.
    centers = [0.5]
    mfs1 = build_triangles([0.5])
    mfs2 = build_triangles([0.3])
    
    # Define two different 2D rule constant arrays (shape 3x3 because we have three MFs per input).
    rule_constants1 = np.array([1.,  0.2, 0.1, 0.8, 0.8, 0.2, 0.6, 0.2, 0.4]).reshape(3, 3)
    
    rule_constants2 = np.array([0,0.1,0.7,0.2,0.3,0.8,0.3,0.6,1]).reshape(3, 3)
    
    # Compute the surfaces.
    X1, X2, Y1 = compute_fis_surface(mfs1, mfs2, rule_constants1)
    _,  _, Y2 = compute_fis_surface(mfs1, mfs2, rule_constants2)
    
    # Create a single figure with two subplots.
    fig = plt.figure(figsize=(12, 6))
    
    # Surface 1.
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X1, X2, Y1, cmap='viridis')
    ax1.set_xlabel('Input 1')
    ax1.set_ylabel('Input 2')
    ax1.set_zlabel('FIS Output')
    ax1.set_title('FIS Surface 1')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # Surface 2.
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X1, X2, Y2, cmap='viridis')
    ax2.set_xlabel('Input 1')
    ax2.set_ylabel('Input 2')
    ax2.set_zlabel('FIS Output')
    ax2.set_title('FIS Surface 2')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_2d_fis_surfaces()
