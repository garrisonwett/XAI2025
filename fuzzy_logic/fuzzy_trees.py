import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EPS = np.finfo(float).eps

def triangular_mf(x, a, b, c):
    x_arr = np.array(x, dtype=float)
    mu = np.zeros_like(x_arr)
    left_mask  = (x_arr > a) & (x_arr < b)
    right_mask = (x_arr >= b) & (x_arr < c)
    mu[left_mask]  = (x_arr[left_mask] - a) / (b - a)
    mu[right_mask] = (c - x_arr[right_mask]) / (c - b)
    return mu.item() if np.isscalar(x) else mu

def build_triangles(centers):
    sorted_centers = np.sort(centers)
    full = np.concatenate(([0.0], sorted_centers, [1.0]))
    mfs = []
    n = len(full)
    for i in range(n):
        left   = full[i-1] if i>0     else full[0]
        center = full[i]
        right  = full[i+1] if i<n-1   else full[-1]
        mfs.append(lambda x, l=left, c=center, r=right: triangular_mf(x, l, c, r))
    return mfs

def tsk_inference_const(x1, x2, x1_mfs, x2_mfs, rule_constants):
    num = 0.0
    den = 0.0
    for i, mf1 in enumerate(x1_mfs):
        for j, mf2 in enumerate(x2_mfs):
            w = mf1(x1) * mf2(x2)
            y_ij = rule_constants[i, j]
            num += w * y_ij
            den += w
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
