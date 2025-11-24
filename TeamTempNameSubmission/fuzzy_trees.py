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

    Assumes a < b < c. At x == b the degree is 1, at x <= a or x >= c the degree is 0.
    """
    # Guard against degenerate intervals
    if c <= a:
        return 0.0

    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0

    # Left side
    if a < x < b:
        if b == a:
            return 0.0
        return (x - a) / (b - a)

    # Right side
    if b < x < c:
        if c == b:
            return 0.0
        return (c - x) / (c - b)

    # Any remaining numerical edge case
    return 0.0


def build_triangles(centers):
    """
    Builds triangular membership functions over [0,1] given an array of centers.

    For sorted centers c0 < c1 < ... < c_(n-1), we create n membership functions:

        MF_0:   peak at c0
            left foot  at 0.0
            right foot at (c0 + c1) / 2

        MF_i (0 < i < n-1): peak at c_i
            left foot  at (c_(i-1) + c_i) / 2
            right foot at (c_i + c_(i+1)) / 2

        MF_(n-1): peak at c_(n-1)
            left foot  at (c_(n-2) + c_(n-1)) / 2
            right foot at 1.0

    This gives a smooth, overlapping partition of [0,1] with well behaved edges.
    """
    if not centers:
        return []

    sorted_centers = sorted(centers)

    # Clamp centers to [0,1]
    sorted_centers = [min(max(c, 0.0), 1.0) for c in sorted_centers]

    mfs = []
    n = len(sorted_centers)

    for i, center in enumerate(sorted_centers):
        if i == 0:
            # Leftmost MF
            left = 0.0
            if n == 1:
                right = 1.0
            else:
                right = 0.5 * (sorted_centers[0] + sorted_centers[1])
        elif i == n - 1:
            # Rightmost MF
            left = 0.5 * (sorted_centers[n - 2] + sorted_centers[n - 1])
            right = 1.0
        else:
            # Interior MF
            left = 0.5 * (sorted_centers[i - 1] + sorted_centers[i])
            right = 0.5 * (sorted_centers[i] + sorted_centers[i + 1])

        # Freeze the current values using default arguments in the lambda.
        mf = lambda x, a=left, b=center, c=right: triangular_mf(x, a, b, c)
        mfs.append(mf)

    return mfs


# -------------------------------------------------------------------------
#             TSK Inference Functions
# -------------------------------------------------------------------------

def _parse_params_entry(entry, mode):
    """
    Handle params[i][j] entries of length 2 or 3.

    For additive mode:
        len == 2: [p1, p2] -> p0 = 0.0, p1, p2
        len >= 3: [p0, p1, p2, ...] -> p0, p1, p2

    For multiplicative mode:
        len == 2: [p1, p2]
        len >= 3: [p0, p1, p2, ...] -> ignore p0, keep p1, p2
    """
    if mode == "add":
        if len(entry) == 2:
            p1, p2 = entry
            p0 = 0.0
        else:
            p0, p1, p2 = entry[:3]
        return p0, p1, p2
    else:
        # multiplicative mode
        if len(entry) == 2:
            p1, p2 = entry
        else:
            _, p1, p2 = entry[:3]
        return p1, p2


def _tsk_inference_core(x1, x2, x1_mfs, x2_mfs, params, mode="add"):
    """
    Two input TSK inference with two possible modes:

        mode="mult":
            y_ij = p1 * x1 * p2 * x2
            params[i][j] is [p1, p2] or [p0, p1, p2] (p0 ignored)

        mode="add":
            y_ij = p0 + p1 * x1 + p2 * x2
            params[i][j] is [p1, p2] or [p0, p1, p2]

    The final output is the weighted average:

        y = sum_ij w_ij * y_ij / sum_ij w_ij
        where w_ij = mu1_i(x1) * mu2_j(x2)
    """
    numerator = 0.0
    denominator = 0.0

    # Precompute membership degrees once per input
    x1_mus = [mf(x1) for mf in x1_mfs]
    x2_mus = [mf(x2) for mf in x2_mfs]

    for i, mu1 in enumerate(x1_mus):
        if mu1 == 0.0:
            continue
        for j, mu2 in enumerate(x2_mus):
            if mu2 == 0.0:
                continue

            w_ij = mu1 * mu2  # Rule firing strength
            entry = params[i][j]

            if mode == "mult":
                p1, p2 = _parse_params_entry(entry, mode="mult")
                y_ij = p1 * x1 * p2 * x2
            else:
                p0, p1, p2 = _parse_params_entry(entry, mode="add")
                y_ij = p0 + p1 * x1 + p2 * x2

            numerator += w_ij * y_ij
            denominator += w_ij

    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def tsk_inference_mult(x1, x2, x1_mfs, x2_mfs, params):
    """
    Multiplicative TSK:
        y_ij = p1 * x1 * p2 * x2

    params[i][j] can be:
        [p1, p2]          (original form)
        [p0, p1, p2, ...] (p0 ignored)
    """
    return _tsk_inference_core(x1, x2, x1_mfs, x2_mfs, params, mode="mult")


def tsk_inference_add(x1, x2, x1_mfs, x2_mfs, params):
    """
    Additive affine TSK:
        y_ij = p0 + p1 * x1 + p2 * x2

    params[i][j] can be:
        [p1, p2]          -> p0 = 0
        [p0, p1, p2, ...] -> p0, p1, p2
    """
    return _tsk_inference_core(x1, x2, x1_mfs, x2_mfs, params, mode="add")


# Main API used by the controller
tsk_inference = tsk_inference_add

# Backward compatible alias in case older code uses this name
tsk_inference_const = tsk_inference


# -------------------------------------------------------------------------
#                        Visualization Functions
# -------------------------------------------------------------------------

def plot_mfs(mfs, x_range=(0, 1), resolution=1000, title="Membership Functions"):
    """
    Plots a set of membership functions over the specified x_range.
    """
    x_values = np.linspace(x_range[0], x_range[1], resolution)
    plt.figure(figsize=(6, 4))
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


def _tsk_surface_grid_fast(x1_vals, x2_vals, x1_mfs, x2_mfs, params, mode="add"):
    """
    Fast vectorized computation of the TSK surface for two inputs.

    For mode="mult":
        y_ij = p1 * x1 * p2 * x2

    For mode="add":
        y_ij = p0 + p1 * x1 + p2 * x2
    """
    n1 = len(x1_mfs)
    n2 = len(x2_mfs)
    M = len(x1_vals)
    N = len(x2_vals)

    # Membership degrees for all mfs and grid points
    mu1 = np.empty((n1, M))
    mu2 = np.empty((n2, N))
    for i in range(n1):
        mu1[i, :] = [x1_mfs[i](x) for x in x1_vals]
    for j in range(n2):
        mu2[j, :] = [x2_mfs[j](y) for y in x2_vals]

    # Denominator: sum_i mu1_i(x1) * sum_j mu2_j(x2)
    sum_mu1 = mu1.sum(axis=0)          # shape (M,)
    sum_mu2 = mu2.sum(axis=0)          # shape (N,)
    denom_grid = np.outer(sum_mu1, sum_mu2)  # shape (M, N)

    # Build grids for x1, x2 (shape (M, N))
    X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing="ij")

    if mode == "mult":
        # Precompute p1 * p2 for each rule
        P = np.empty((n1, n2))
        for i in range(n1):
            for j in range(n2):
                p1, p2 = _parse_params_entry(params[i][j], mode="mult")
                P[i, j] = p1 * p2

        # F[k, l] = sum_{i,j} P[i,j] * mu1[i,k] * mu2[j,l]
        F = np.einsum("ij,ik,jl->kl", P, mu1, mu2)  # shape (M, N)

        # Numerator: x1 * x2 * F
        num_grid = X1 * X2 * F

    else:
        # Additive affine case: y_ij = p0 + p1 * x1 + p2 * x2
        P0 = np.empty((n1, n2))
        P1 = np.empty((n1, n2))
        P2 = np.empty((n1, n2))
        for i in range(n1):
            for j in range(n2):
                p0, p1, p2 = _parse_params_entry(params[i][j], mode="add")
                P0[i, j] = p0
                P1[i, j] = p1
                P2[i, j] = p2

        # A[k, l] = sum_{i,j} P0[i,j] * mu1[i,k] * mu2[j,l]
        A = np.einsum("ij,ik,jl->kl", P0, mu1, mu2)  # shape (M, N)
        # B[k, l] = sum_{i,j} P1[i,j] * mu1[i,k] * mu2[j,l]
        B = np.einsum("ij,ik,jl->kl", P1, mu1, mu2)
        # C[k, l] = sum_{i,j} P2[i,j] * mu1[i,k] * mu2[j,l]
        C = np.einsum("ij,ik,jl->kl", P2, mu1, mu2)

        num_grid = A + X1 * B + X2 * C

    Z = np.zeros_like(num_grid)
    mask = denom_grid != 0.0
    Z[mask] = num_grid[mask] / denom_grid[mask]

    return Z


def plot_tsk_surface(x1_mfs, x2_mfs, params, resolution=50, mode="add"):
    """
    Plots the TSK output surface y = f(x1, x2) in 3D.

    mode="add" uses the additive affine TSK.
    mode="mult" uses the multiplicative TSK.
    """
    x1_vals = np.linspace(0.000001, 0.999999, resolution)
    x2_vals = np.linspace(0.000001, 0.999999, resolution)

    Z = _tsk_surface_grid_fast(x1_vals, x2_vals, x1_mfs, x2_mfs, params, mode=mode)

    X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing="ij")

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("TSK Output")
    ax.set_title(f"TSK Output Surface ({mode} mode)")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


# -------------------------------------------------------------------------
#                               Main Demo
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple demo to visually verify things
    az_centers = [0.25, 0.5, 0.75]
    closure_centers = [0.3, 0.6]

    x1_mfs = build_triangles(az_centers)
    x2_mfs = build_triangles(closure_centers)

    plot_mfs(x1_mfs, x_range=(0, 1), title="x1 Membership Functions")
    plot_mfs(x2_mfs, x_range=(0, 1), title="x2 Membership Functions")

    num_rules_x1 = len(x1_mfs)
    num_rules_x2 = len(x2_mfs)
    params = []
    for i in range(num_rules_x1):
        row = []
        for j in range(num_rules_x2):
            p0 = 0.1 * (i + j)
            p1 = 1.0 / (1 + i)
            p2 = 1.0 / (1 + j)
            row.append([p0, p1, p2])
        params.append(row)

    plot_tsk_surface(x1_mfs, x2_mfs, params, resolution=25, mode="add")

    test_points = [
        (0.0, 0.0),
        (0.1, 0.4),
        (0.3, 0.5),
        (0.9, 0.9),
        (1.0, 1.0),
    ]

    print("TSK outputs at sample points (additive mode):\n")
    for (x1, x2) in test_points:
        y_out = tsk_inference(x1, x2, x1_mfs, x2_mfs, params)
        print(f"x1={x1:.2f}, x2={x2:.2f} => y={y_out:.3f}")
