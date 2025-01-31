import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import time

t = time.time()
def triangular_mf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    else:
        return 1.0

def build_5_triangles(centers):
    """
    Build 5 triangular MFs on [0,1].
    centers: array of length 3 (e.g. [0.3, 0.5, 0.8]).
    """
    if len(centers) != 3:
        raise ValueError("centers must be an array of length 3.")

    c0 = 0.0
    c1, c2, c3 = centers
    c4 = 1.0
    c = [c0, c1, c2, c3, c4]

    mfs = []
    for i in range(5):
        peak = c[i]
        if i == 0:
            left = c[0]
        else:
            left = 0.5 * (c[i-1] + c[i])

        if i == 4:
            right = c[4]
        else:
            right = 0.5 * (c[i] + c[i+1])

        mf = lambda x, a=left, b=peak, cc=right: triangular_mf(x, a, b, cc)
        mfs.append(mf)

    return mfs

def tsk_inference(x1, x2, x1_mfs, x2_mfs):
    """
    TSK inference with 5 MFs for x1, 5 MFs for x2 => 25 rules.
    Rule consequent here is: y_ij = i + j (trivial).
    """
    numerator = 0.0
    denominator = 0.0

    for i in range(5):
        for j in range(5):
            w_ij = x1_mfs[i](x1) * x2_mfs[j](x2)  # firing strength
            y_ij = i + j                        # trivial consequent
            numerator   += w_ij * y_ij
            denominator += w_ij

    if denominator == 0:
        return 0.0
    return numerator / denominator


# --------------------------------------------------------------------------
#                          VISUALIZATION FUNCTIONS
# --------------------------------------------------------------------------

def plot_mfs(mfs, x_range=(0,1), resolution=100, title="Membership Functions"):
    """
    Plots a set of membership functions (mfs) over x_range.
    mfs: list of callable mfs, each mf(x) -> membership value in [0,1].
    """
    x_values = np.linspace(x_range[0], x_range[1], resolution)

    plt.figure(figsize=(6,4))
    for i, mf in enumerate(mfs):
        y_values = [mf(x) for x in x_values]
        plt.plot(x_values, y_values, label=f"MF {i}")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Membership")
    plt.ylim([-0.05, 1.05])
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_tsk_surface(x1_mfs, x2_mfs, resolution=50):
    """
    Plots the TSK output surface y = f(x1, x2) in 3D,
    given two lists of 5 membership functions each (x1_mfs, x2_mfs).
    
    resolution: number of steps in [0,1] for x1, x2.
    """
    x1_vals = np.linspace(0, 1, resolution)
    x2_vals = np.linspace(0, 1, resolution)
    Z = np.zeros((resolution, resolution))

    # Compute TSK output over a grid in (x1, x2)
    for i, xv in enumerate(x1_vals):
        for j, yv in enumerate(x2_vals):
            Z[j, i] = tsk_inference(xv, yv, x1_mfs, x2_mfs)
    
    # Create 2D meshgrid for plotting
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # --- 3D Surface Plot ---
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('TSK Output')
    ax.set_title("TSK Output Surface")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # --- (Optional) 2D Contour Plot ---
    # If you also want a contour view, uncomment below:
    """
    plt.figure(figsize=(6,4))
    contour = plt.contourf(X1, X2, Z, cmap='viridis', levels=25)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("TSK Output Contour")
    plt.colorbar(contour)
    plt.show()
    """


# --------------------------------------------------------------------------
#                          DEMO / MAIN
# --------------------------------------------------------------------------

if __name__ == "__main__":
    centers = [0.3, 0.5, 0.8]
    x1_mfs = build_5_triangles(centers)
    x2_mfs = build_5_triangles(centers)

    # Plot membership functions for x1 and x2
    plot_mfs(x1_mfs, x_range=(0,1), title="x1 Membership Functions")
    plot_mfs(x2_mfs, x_range=(0,1), title="x2 Membership Functions")

    # Show how the TSK output changes across x1,x2
    plot_tsk_surface(x1_mfs, x2_mfs, resolution=50)

    t = time.time()
    # Test some points in [0,1]
    test_points = [(0.0, 0.0),
                   (0.1, 0.4),
                   (0.3, 0.5),
                   (0.6, 0.9),
                   (1.0, 1.0)]
    
    print("Some TSK outputs at discrete points:\n")
    for (x1, x2) in test_points:
        y_out = tsk_inference(x1, x2, x1_mfs, x2_mfs)
        print(f"x1={x1:.2f}, x2={x2:.2f} => y={y_out:.3f}")


    print("Elapsed time:", time.time() - t)