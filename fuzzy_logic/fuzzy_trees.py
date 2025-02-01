import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# -------------------------------------------------------------------------
#           Membership Function and Flexible Builder
# -------------------------------------------------------------------------

def triangular_mf(x, a, b, c):
    """
    Triangular membership function with feet at a and c and peak at b.
    Returns the membership degree in [0,1] for a given x.
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    else:
        return 1.0

def build_triangles(centers):
    """
    Builds triangular membership functions over [0,1] given an array of middle centers.
    
    The membership functions are centered at:
        0, (sorted centers...), 1.
    
    For each center, the left boundary is the midpoint with the previous center,
    and the right boundary is the midpoint with the next center (with 0 and 1 clamped).
    
    Parameters:
        centers (list or array): A list of center values (should be between 0 and 1).
                                  They do not need to be sorted.
    
    Returns:
        mfs (list): A list of membership functions (each is a callable function mf(x)).
    """
    # Ensure the centers are sorted
    sorted_centers = sorted(centers)
    # Include endpoints 0 and 1
    full_centers = [0.0] + sorted_centers + [1.0]
    
    mfs = []
    for i in range(len(full_centers)):
        center = full_centers[i]
        if i == 0:
            left = full_centers[0]  # 0.0
        else:
            left = full_centers[i-1]
        if i == len(full_centers) - 1:
            right = full_centers[-1]  # 1.0
        else:
            right = full_centers[i+1]
        # Freeze the current values using default arguments in the lambda.
        mf = lambda x, a=left, b=center, c=right: triangular_mf(x, a, b, c)
        mfs.append(mf)
    return mfs

# -------------------------------------------------------------------------
#             Realistic TSK Inference Function
# -------------------------------------------------------------------------

def tsk_inference(x1, x2, x1_mfs, x2_mfs, params):
    """
    Computes the TSK output for inputs x1 and x2 using a rule base defined by the provided
    membership functions and parameter matrix.
    
    Each rule (for indices i, j) has a consequent of the form:
        y_ij = p0 + p1*x1 + p2*x2
    where params[i][j] = [p0, p1, p2].
    
    The overall output is the weighted average of the rule outputs, with weights equal to the
    product of the corresponding membership degrees.
    """
    numerator = 0.0
    denominator = 0.0

    # Loop over all membership functions for x1 and x2 (the rule base)
    for i in range(len(x1_mfs)):
        for j in range(len(x2_mfs)):
            w_ij = x1_mfs[i](x1) * x2_mfs[j](x2)  # Rule firing strength
            p1, p2 = params[i][j]
            y_ij = -1*(p1 * abs(x1-0.5)-0.25) * p2 * x2        # Linear consequent
            numerator   += w_ij * y_ij
            denominator += w_ij

    if denominator == 0:
        return 0.0
    return 700*numerator / denominator

# -------------------------------------------------------------------------
#                        Visualization Functions
# -------------------------------------------------------------------------

def plot_mfs(mfs, x_range=(0,1), resolution=1000, title="Membership Functions"):
    """
    Plots a set of membership functions over the specified x_range.
    
    Parameters:
        mfs: List of membership functions.
        x_range: Tuple (min, max) for the x-axis.
        resolution: Number of points to sample in x_range.
        title: Title of the plot.
    """
    x_values = np.linspace(x_range[0], x_range[1], resolution)
    plt.figure(figsize=(6,4))
    for i, mf in enumerate(mfs):
        y_values = [mf(x) for x in x_values]
        plt.plot(x_values, y_values, label=f"MF {i}")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Membership Degree")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_tsk_surface(x1_mfs, x2_mfs, params, resolution=50):
    """
    Plots the TSK output surface y = f(x1, x2) in 3D.
    
    Parameters:
        x1_mfs: List of membership functions for x1.
        x2_mfs: List of membership functions for x2.
        params: Parameter matrix for the rule consequents.
        resolution: Number of grid points in [0,1] for x1 and x2.
    """
    x1_vals = np.linspace(0.000001, 0.999999, resolution)
    x2_vals = np.linspace(0.000001, 0.99999, resolution)
    Z = np.zeros((resolution, resolution))

    # Compute TSK output over the grid
    for i, xv in enumerate(x1_vals):
        for j, yv in enumerate(x2_vals):
            Z[j, i] = tsk_inference(xv, yv, x1_mfs, x2_mfs, params)

    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # 3D Surface Plot
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Az')
    ax.set_ylabel('Closure')
    ax.set_zlabel('TSK Output')
    ax.set_title("TSK Output Surface")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # (Optional) 2D Contour Plot:
    """
    plt.figure(figsize=(6,4))
    contour = plt.contourf(X1, X2, Z, cmap='viridis', levels=25)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("TSK Output Contour")
    plt.colorbar(contour)
    plt.show()
    """

# -------------------------------------------------------------------------
#                               Main Demo
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # Define your "middle" centers (they don't need to be fixed in number)
    az_centers = [0.5] 
    closure_centers = [0.5]
    

    # You could also try: centers = [0.2, 0.4, 0.6, 0.9] (which yields 6 MFs)

    # Build membership functions for x1 and x2 based on the provided centers.
    x1_mfs = build_triangles(az_centers)
    x2_mfs = build_triangles(closure_centers)

    # Visualize the membership functions
    # plot_mfs(x1_mfs, x_range=(0,1), title="x1 Membership Functions")
    # plot_mfs(x2_mfs, x_range=(0,1), title="x2 Membership Functions")

    # Set up realistic parameters for the TSK rule consequents.
    # For each rule, we assume a linear consequent: y = p0 + p1*x1 + p2*x2.
    # Since the number of rules equals len(x1_mfs) x len(x2_mfs),
    # we create a parameter matrix accordingly.
    num_rules_x1 = len(x1_mfs)
    num_rules_x2 = len(x2_mfs)
    params = []
    for i in range(num_rules_x1):

        row = []
        for j in range(num_rules_x2):
            # Example: p0, p1, and p2 are chosen based on the rule indices.
            p1 = 1 
            p2 = max(j-1,0)
            row.append([p1, p2])
        params.append(row)

    # Visualize the TSK output surface in 3D.
    plot_tsk_surface(x1_mfs, x2_mfs, params, resolution=50)

    # Test the TSK system at some discrete points.
    test_points = [(0.0, 0.0),
                   (0.1, 0.4),
                   (0.3, 0.5),
                   (0.6, 0.9),
                   (1.0, 1.0)]
    
    print("TSK outputs at sample points:\n")
    for (x1, x2) in test_points:
        y_out = tsk_inference(x1, x2, x1_mfs, x2_mfs, params)
        print(f"x1={x1:.2f}, x2={x2:.2f} => y={y_out:.3f}")
