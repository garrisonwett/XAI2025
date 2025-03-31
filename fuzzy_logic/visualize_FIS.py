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

# 4D TSK inference (constant output) using four inputs and four sets of MFs.
def tsk_inference_const_4d(x1, x2, x3, x4, mfs1, mfs2, mfs3, mfs4, rule_constants):
    num = 0.0
    den = 0.0
    for i, mf1 in enumerate(mfs1):
        for j, mf2 in enumerate(mfs2):
            for k, mf3 in enumerate(mfs3):
                for l, mf4 in enumerate(mfs4):
                    w = mf1(x1) * mf2(x2) * mf3(x3) * mf4(x4)
                    y = rule_constants[i, j, k, l]
                    num += w * y
                    den += w
    return num / (den + EPS)

# For a fixed value of input 1, create a 3D scatter plot of outputs over inputs 2, 3, and 4.
def plot_tsk_output_3d_for_fixed_input1(fixed_x1, mfs1, mfs2, mfs3, mfs4, rule_constants, resolution=10):
    # Generate grids for inputs 2, 3, and 4.
    x2_vals = np.linspace(EPS, 1 - EPS, resolution)
    x3_vals = np.linspace(EPS, 1 - EPS, resolution)
    x4_vals = np.linspace(EPS, 1 - EPS, resolution)
    
    X2, X3, X4 = np.meshgrid(x2_vals, x3_vals, x4_vals, indexing='ij')
    # Flatten the grid arrays for iteration.
    X2_flat = X2.flatten()
    X3_flat = X3.flatten()
    X4_flat = X4.flatten()
    outputs = np.zeros_like(X2_flat)
    
    # Compute the fuzzy output for every combination of inputs 2, 3, and 4.
    for idx, (x2, x3, x4) in enumerate(zip(X2_flat, X3_flat, X4_flat)):
        outputs[idx] = tsk_inference_const_4d(fixed_x1, x2, x3, x4, mfs1, mfs2, mfs3, mfs4, rule_constants)
    
    # Create a 3D scatter plot with color representing the output.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X2_flat, X3_flat, X4_flat, c=outputs, cmap='viridis')
    ax.set_xlabel('Input 2')
    ax.set_ylabel('Input 3')
    ax.set_zlabel('Input 4')
    ax.set_title(f'TSK Output (fixed Input 1 = {fixed_x1:.2f})')
    fig.colorbar(scatter, ax=ax, label='TSK Output')
    plt.tight_layout()

# Main visualization routine: for several fixed x1 values, plot the outputs.
def visualize_4d_tsk_outputs():
    # For demonstration, use the same center for all inputs.
    centers = [0.5]
    mfs1 = build_triangles(centers)
    mfs2 = build_triangles(centers)
    mfs3 = build_triangles(centers)
    mfs4 = build_triangles(centers)
    
    # Create a sample 4D rule constant array with shape (3,3,3,3).
    rule_constants = np.arange(81).reshape(3, 3, 3, 3)
    
    # Choose several fixed values for input 1.
    fixed_x1_values = [0.25, 0.5, 0.75]
    
    for fixed_x1 in fixed_x1_values:
        plot_tsk_output_3d_for_fixed_input1(fixed_x1, mfs1, mfs2, mfs3, mfs4, rule_constants, resolution=5)
    
    plt.show()

if __name__ == "__main__":
    visualize_4d_tsk_outputs()
