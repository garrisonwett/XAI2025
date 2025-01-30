import numpy as np
import matplotlib.pyplot as plt
import time

def triangular_mf(x, a, b, c):
    """
    Returns the membership degree of x in a triangular fuzzy set 
    with 'feet' at a and c and peak at b.
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    else:
        # covers x == b exactly
        return 1.0

def build_5_triangles(centers):
    """
    Builds 5 triangular MFs on [0,1].
    
    centers: array-like of length 3, e.g. [0.3, 0.5, 0.8]
    
    The 5 centers will be:
      c0 = 0.0
      c1 = centers[0]
      c2 = centers[1]
      c3 = centers[2]
      c4 = 1.0
      
    Each triangle i uses:
      peak = c[i]
      left = midpoint(c[i-1], c[i])  (or clamped to 0 if i=0)
      right = midpoint(c[i], c[i+1]) (or clamped to 1 if i=4)
    """
    if len(centers) != 3:
        raise ValueError("centers must be an array of length 3.")

    c0 = 0.0
    c1, c2, c3 = centers
    c4 = 1.0

    # list of centers
    c = [c0, c1, c2, c3, c4]

    mfs = []
    for i in range(5):
        peak = c[i]
        if i == 0:
            left = c[0]  # clamp to 0
        else:
            left = 0.5 * (c[i-1] + c[i])  # midpoint of c[i-1] and c[i]

        if i == 4:
            right = c[4] # clamp to 1
        else:
            right = 0.5 * (c[i] + c[i+1]) # midpoint of c[i] and c[i+1]

        # create a small lambda capturing the parameters
        mf = lambda x, a=left, b=peak, cc=right: triangular_mf(x, a, b, cc)
        mfs.append(mf)

    return mfs

def plot_mfs(mfs, resolution=100000, x_range=(0, 1), title="Membership Functions"):
    """
    Plots the membership functions in mfs using matplotlib.

    Parameters:
    - mfs: list of membership functions (each mf is a callable: mf(x)->membership)
    - resolution: number of points used to sample each MF
    - x_range: the (min, max) range on the x-axis to sample
    - title: plot title
    """
    x_vals = np.linspace(x_range[0], x_range[1], resolution)

    plt.figure(figsize=(7,4))
    for i, mf in enumerate(mfs):
        y_vals = [mf(x) for x in x_vals]
        plt.plot(x_vals, y_vals, label=f"MF {i}")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Membership degree")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()

def tsk_inference(x1, x2, x1_mfs, x2_mfs):
    """
    Example TSK inference with 5 MFs for x1 and 5 for x2 => 25 rules.

    Rule consequent is just y_ij = i + j for demonstration.
    Weighted average used for final output.
    """
    numerator = 0.0
    denominator = 0.0

    # 5 MFs for x1, 5 MFs for x2 => 25 possible rule combinations
    for i in range(5):
        for j in range(5):
            w_ij = x1_mfs[i](x1) * x2_mfs[j](x2)  # firing strength
            y_ij = i + j                        # trivial consequent
            numerator   += w_ij * y_ij
            denominator += w_ij

    if denominator == 0:
        return 0.0
    return numerator / denominator

if __name__ == "__main__":
    t = time.perf_counter()
    # Example: define the 3 middle centers in [0,1].
    centers = [0.25, 0.5, 0.75]

    # Build membership functions for x1 and x2
    x1_mfs = build_5_triangles(centers)
    x2_mfs = build_5_triangles(centers)

    # ---- VISUALIZE MEMBERSHIP FUNCTIONS ----
    plot_mfs(x1_mfs, x_range=(0,1), title="x1 MFs")
    plot_mfs(x2_mfs, x_range=(0,1), title="x2 MFs")

    # ---- TEST THE TSK INFERENCE ----

    # Test points
    test_points = []
    for i in range(100):
        for j in range(100):
            test_points.append((i/100, j/100))

    for (xx1, xx2) in test_points:
        y_out = tsk_inference(xx1, xx2, x1_mfs, x2_mfs)
        print(f"x1={xx1:.2f}, x2={xx2:.2f} => y={y_out:.3f}")
    print(f"Time taken: {time.perf_counter() - t:.6f} seconds")
