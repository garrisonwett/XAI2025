import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable

# --------------------------------------------------------------
# 1. Triangular membership function
# --------------------------------------------------------------
def triangular_mf_vectorized(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if a == b and x == a:
        return 1.0
    if b == c and x == c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


# --------------------------------------------------------------
# 2. Build three triangles per input with sum-to-one constraint
# --------------------------------------------------------------
def build_three_triangles(center_val: float):
    center_val = float(np.clip(center_val, 0.0, 1.0))

    # Triangles: left, center, right
    t0 = (0.0, 0.0, center_val if center_val > 0 else 1e-6)
    t1 = (0.0, center_val, 1.0)
    t2 = (center_val if center_val < 1 else 1 - 1e-6, 1.0, 1.0)

    triangles = [t0, t1, t2]

    # Correct lambda creation. Each MF is directly a function.
    mf_list = [
        (lambda a=a, b=b, c=c: (lambda x, aa=a, bb=b, cc=c:
            triangular_mf_vectorized(x, aa, bb, cc)))
        ()  # Call the wrapper immediately to get the function
        for (a, b, c) in triangles
    ]

    return mf_list, triangles


def build_all_input_mfs(center_values: List[float]):
    mf_sets = []
    triangle_sets = []
    for c in center_values:
        mfs, tris = build_three_triangles(c)
        mf_sets.append(mfs)
        triangle_sets.append(tris)
    return mf_sets, triangle_sets


# --------------------------------------------------------------
# 3. TSK Rule object
# --------------------------------------------------------------
class TSKRule:
    def __init__(self, mf_indices: Tuple[int], coeffs: np.ndarray):
        self.mf_indices = mf_indices
        self.coeffs = coeffs


# --------------------------------------------------------------
# 4. Build TSK tree
# --------------------------------------------------------------
def build_tsk_tree(mf_sets: List[List[Callable]], coeff_matrix: np.ndarray):
    from itertools import product
    rule_indices = list(product(*[range(3) for _ in mf_sets]))
    rules = [
        TSKRule(mf_indices=idx_tuple, coeffs=coeff_matrix[i])
        for i, idx_tuple in enumerate(rule_indices)
    ]
    return rules


# --------------------------------------------------------------
# 5. Rule evaluation
# --------------------------------------------------------------
def evaluate_rule(rule: TSKRule, mf_sets: List[List[Callable]], inputs: np.ndarray):
    mu = 1.0
    for dim, mf_idx in enumerate(rule.mf_indices):
        mf = mf_sets[dim][mf_idx]  # This must be a function
        mu *= mf(inputs[dim])      # Now this is safe

    y = rule.coeffs[0] + np.dot(rule.coeffs[1:], inputs)
    return mu, y


# --------------------------------------------------------------
# 6. Full TSK inference
# --------------------------------------------------------------
def tsk_tree_inference(rules: List[TSKRule], mf_sets: List[List[Callable]], inputs: np.ndarray):
    wsum = 0.0
    ysum = 0.0

    for rule in rules:
        mu, y = evaluate_rule(rule, mf_sets, inputs)
        if mu > 0.0:
            wsum += mu
            ysum += mu * y

    if wsum == 0.0:
        return 0.0
    return ysum / wsum


# --------------------------------------------------------------
# 7. Visualization for all MFs
# --------------------------------------------------------------
def visualize_membership_functions(triangle_sets: List[List[Tuple[float, float, float]]],
                                   resolution: int = 500):
    x = np.linspace(0, 1, resolution)

    num_inputs = len(triangle_sets)
    fig, axes = plt.subplots(num_inputs, 1, figsize=(8, 3 * num_inputs))

    if num_inputs == 1:
        axes = [axes]

    for idx, (triangles, ax) in enumerate(zip(triangle_sets, axes)):
        for t in triangles:
            a, b, c = t
            y = np.array([triangular_mf_vectorized(val, a, b, c) for val in x])
            ax.plot(x, y)

        ax.set_title(f"Input {idx} Membership Functions")
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlim([0, 1])
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# Example usage
# --------------------------------------------------------------
if __name__ == "__main__":
    center_values = [0.3, 0.8]

    mf_sets, triangle_sets = build_all_input_mfs(center_values)
    visualize_membership_functions(triangle_sets)

    num_inputs = len(center_values)
    num_rules = 3 ** num_inputs
    coeff_matrix = np.random.uniform(-1, 1, size=(num_rules, num_inputs + 1))

    rules = build_tsk_tree(mf_sets, coeff_matrix)

    test_input = np.array([0.4, 0.9])
    output = tsk_tree_inference(rules, mf_sets, test_input)
    print("TSK output:", output)
