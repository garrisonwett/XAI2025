import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numba


EPS = np.finfo(float).eps

def triangular_mf(x, a, b, c):
    # For scalar x, avoid array overhead.
    if np.isscalar(x):
        if x <= a or x >= c:
            return 0.0
        elif x < b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    else:
        # Ensure x is a NumPy array (without needless copying if possible).
        x_arr = np.asarray(x, dtype=float)
        # Compute the two linear segments and use np.minimum/np.maximum to get the triangular shape.
        y = np.maximum(0, np.minimum((x_arr - a) / (b - a), (c - x_arr) / (c - b)))
        return y

def build_triangles(centers):
    centers = np.asarray(centers)
    sorted_centers = np.sort(centers)
    # Include endpoints 0 and 1.
    full = np.concatenate(([0.0], sorted_centers, [1.0]))
    n = len(full)
    # Precompute left, center, and right for each triangle.
    left = np.empty(n)
    center = full.copy()
    right = np.empty(n)
    left[0] = full[0]
    left[1:] = full[:-1]
    right[:-1] = full[1:]
    right[-1] = full[-1]
    # Return a list of lambda functions; capture the parameters in default arguments to avoid repeated lookups.
    mfs = [lambda x, l=left[i], c=center[i], r=right[i]: triangular_mf(x, l, c, r)
           for i in range(n)]
    return mfs

def tsk_inference_const(x1, x2, x1_mfs, x2_mfs, rule_constants):
    # Compute membership values for each input.
    w1 = np.array([mf(x1) for mf in x1_mfs])
    w2 = np.array([mf(x2) for mf in x2_mfs])
    # Use dot products to compute numerator and denominator without forming an outer product.
    num = w1.dot(rule_constants).dot(w2)
    den = w1.sum() * w2.sum()
    return num / (den + EPS)

def plot_mfs(mfs, title_str):
    x = np.linspace(0, 1, 1000)
    plt.figure()
    for idx, mf in enumerate(mfs, 1):
        plt.plot(x, mf(x), label=f"MF {idx}")
    plt.title(title_str)
    plt.xlabel('x'); plt.ylabel('Membership Degree')
    plt.legend(); plt.grid(True)
    plt.ylim(0,1)

def plot_tsk_surface(x1_mfs, x2_mfs, rule_constants, resolution=50):
    x1 = np.linspace(EPS, 1-EPS, resolution)
    x2 = np.linspace(EPS, 1-EPS, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(resolution):
        for j in range(resolution):
            Z[j,i] = tsk_inference_const(X1[j,i], X2[j,i], x1_mfs, x2_mfs, rule_constants)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z)
    ax.set_xlabel('X1'); ax.set_ylabel('X2'); ax.set_zlabel('TSK Output')
    ax.set_title('Constant-TSK Output Surface')
    fig.colorbar(surf)

def main():
    x1_centers = np.array([0.5])
    x2_centers = np.array([0.5])

    x1_mfs = build_triangles(x1_centers)
    x2_mfs = build_triangles(x2_centers)

    # Create a simple constant output table (9 rules)
    rule_constants = np.array([0,-100,-500,0,100,500,0,-100,-500]).reshape(len(x1_mfs), len(x2_mfs))
    print("Rule Constants:")
    print(rule_constants)
    plot_mfs(x1_mfs, 'x1 Membership Functions')
    plot_mfs(x2_mfs, 'x2 Membership Functions')
    plot_tsk_surface(x1_mfs, x2_mfs, rule_constants)

    test_points = np.array([[0,0], [0.1,0.4], [0.3,0.5], [0.9,0.9], [1,1]])
    print("TSK outputs at sample points:")
    for x1, x2 in test_points:
        y = tsk_inference_const(x1, x2, x1_mfs, x2_mfs, rule_constants)
        print(f"x1={x1:.2f}, x2={x2:.2f} => y={y:.3f}")

    plt.show()

if __name__ == "__main__":
    main()
