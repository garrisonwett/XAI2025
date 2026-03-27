#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║          FUZZY TREE CHROMOSOME INSPECTOR                               ║
║                                                                        ║
║  Load a saved .pkl chromosome and generate all visualizations /        ║
║  summaries without needing to hand-define anything.                    ║
╚══════════════════════════════════════════════════════════════════════════╝

USAGE
─────
  python inspect_chromosome.py best_chromosome.pkl
  python inspect_chromosome.py path/to/any_chromosome.pkl --no-plots
  python inspect_chromosome.py best_chromosome.pkl --output-dir results/

OPTIONS
───────
  positional arg     Path to the .pkl chromosome file
  --no-plots         Skip generating PNG files (text summary only)
  --no-show          Save PNGs but don't open the interactive plot window
  --output-dir DIR   Directory to save PNGs into (default: current dir)

INPUT REFERENCE  (from redone_controller.py)
────────────────
  Index 0 – Heading alignment   (1 = asteroid ahead, 0 = behind)
  Index 1 – Closure rate         (0 = separating, 1 = approaching fast)
  Index 2 – Asteroid radius      (0 = tiny, 1 = huge)
  Index 3 – Log distance          (0 = very close, 1 = very far)
  Index 4 – Collision urgency    (0 = none, 1 = imminent collision)

RULE TABLE LAYOUT  (9 rules per FIS node)
──────────────────
  Each FIS node takes two inputs (Left child, Right child) and fuzzifies
  each into {Low, Medium, High}.  The 9 rule constants are indexed as:

              Right-child MF →   Low      Med      High
  Left-child MF ↓
        Low                     rule[0]  rule[1]  rule[2]
        Med                     rule[3]  rule[4]  rule[5]
        High                    rule[6]  rule[7]  rule[8]

  Higher rule constant → higher priority score for that input region.
  The output is always a convex combination of these 9 values.
"""

import sys
import os
import pickle
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Import the tree node classes from your project.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from algorithms import (
        InputNode, FISNode, compile_chromosome, clamp_params,
        save_chromosome, copy_tree, get_tree_size, fuzzy_tree_output,
        gather_fis_nodes, gather_leaf_nodes, flatten_tree,
    )
except ImportError:
    print("[WARN] Could not import from algorithms.py — using built-in copies.")
    print("       Place this file next to algorithms.py for full compatibility.\n")

    class InputNode:
        def __init__(self, idx=0):
            self.index = idx
            self.left = self.right = None

    class FISNode:
        def __init__(self):
            self.medium1_center = 0.5
            self.medium2_center = 0.5
            self.rule_constants = [0.0] * 9
            self.left = self.right = None


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

INPUT_NAMES = {
    0: "Heading Alignment",
    1: "Closure Rate",
    2: "Asteroid Radius",
    3: "Log Distance",
    4: "Collision Urgency",
}


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD CHROMOSOME
# ═══════════════════════════════════════════════════════════════════════════

def load_chromosome(pkl_path):
    """Load a chromosome tree from a .pkl file."""
    if not os.path.isfile(pkl_path):
        print(f"ERROR: File not found: {pkl_path}")
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        root = pickle.load(f)

    # Basic sanity check
    if not isinstance(root, FISNode):
        # Some pickles wrap the tree in a list or tuple
        if isinstance(root, (list, tuple)) and len(root) > 0:
            # Try the first element
            candidate = root[0]
            if isinstance(candidate, FISNode):
                root = candidate
            else:
                print(f"ERROR: Loaded object is {type(root)}, not a FISNode tree.")
                print(f"       Contents: {root}")
                sys.exit(1)
        elif isinstance(root, dict) and 'root' in root:
            root = root['root']
        else:
            print(f"ERROR: Loaded object is {type(root)}, not a FISNode tree.")
            sys.exit(1)

    print(f"✓ Loaded chromosome from: {pkl_path}")
    return root


# ═══════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_tree(root, expected_inputs=5):
    """Check that each input 0..expected_inputs-1 appears exactly once."""
    leaves = []
    stack = [root]
    while stack:
        n = stack.pop()
        if isinstance(n, InputNode):
            leaves.append(n.index)
        elif isinstance(n, FISNode):
            assert n.left is not None, "FISNode has no left child!"
            assert n.right is not None, "FISNode has no right child!"
            stack.append(n.left)
            stack.append(n.right)
        else:
            raise TypeError(f"Unknown node type: {type(n)}")

    expected = set(range(expected_inputs))
    found = set(leaves)
    if found != expected:
        missing = expected - found
        extra = found - expected
        msg = []
        if missing:
            msg.append(f"Missing inputs: {sorted(missing)}")
        if extra:
            msg.append(f"Extra/duplicate inputs: {sorted(extra)}")
        if len(leaves) != len(set(leaves)):
            from collections import Counter
            dupes = {k: v for k, v in Counter(leaves).items() if v > 1}
            msg.append(f"Duplicated indices: {dupes}")
        raise ValueError("Tree validation failed! " + "; ".join(msg))

    print(f"✓ Tree valid: {len(leaves)} inputs, "
          f"{sum(1 for _ in _iter_fis(root))} FIS nodes, "
          f"{_count_nodes(root)} total nodes.\n")


def _iter_fis(node):
    """Yield all FISNodes in the tree (BFS order)."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, FISNode):
            yield n
            if n.left: stack.append(n.left)
            if n.right: stack.append(n.right)


def _count_nodes(node):
    count = 0
    stack = [node]
    while stack:
        n = stack.pop()
        count += 1
        if isinstance(n, FISNode):
            if n.left: stack.append(n.left)
            if n.right: stack.append(n.right)
    return count


def _node_label(node):
    """Human-readable label for a node."""
    if isinstance(node, InputNode):
        name = INPUT_NAMES.get(node.index, f"Input {node.index}")
        return f"[{node.index}] {name}"
    elif isinstance(node, FISNode):
        lbl = getattr(node, '_label', None)
        if lbl:
            return lbl.split('\n')[0]
        # Auto-generate a label from children
        left_short = _node_label_short(node.left)
        right_short = _node_label_short(node.right)
        return f"FIS({left_short} × {right_short})"
    return "?"


def _node_label_short(node):
    """Short label for use in auto-generated FIS names."""
    if isinstance(node, InputNode):
        return INPUT_NAMES.get(node.index, f"In{node.index}").split()[0]
    elif isinstance(node, FISNode):
        lbl = getattr(node, '_label', None)
        if lbl:
            return lbl.split('\n')[0].split(':')[0].strip()
        return "FIS"
    return "?"


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def print_tree_summary(root):
    """Print a detailed text description of every node."""
    print("=" * 70)
    print("  TREE SUMMARY")
    print("=" * 70)

    fis_nodes = []
    stack = [root]
    while stack:
        n = stack.pop(0)  # BFS for top-down order
        if isinstance(n, FISNode):
            fis_nodes.append(n)
            if n.left: stack.append(n.left)
            if n.right: stack.append(n.right)

    for i, fis in enumerate(fis_nodes):
        label = getattr(fis, '_label', None) or f"FIS Node {i} ({_node_label(fis)})"
        left_label = _node_label(fis.left)
        right_label = _node_label(fis.right)

        print(f"\n┌─ {label}")
        print(f"│  Left input:  {left_label}   (MF center = {fis.medium1_center:.4f})")
        print(f"│  Right input: {right_label}   (MF center = {fis.medium2_center:.4f})")
        print(f"│")
        print(f"│  Rule Table:")
        print(f"│  {'':>14s}  {'Right_Low':>9s}  {'Right_Med':>9s}  {'Right_Hi':>9s}")
        print(f"│  {'Left_Low':>14s}  {fis.rule_constants[0]:>9.4f}  {fis.rule_constants[1]:>9.4f}  {fis.rule_constants[2]:>9.4f}")
        print(f"│  {'Left_Med':>14s}  {fis.rule_constants[3]:>9.4f}  {fis.rule_constants[4]:>9.4f}  {fis.rule_constants[5]:>9.4f}")
        print(f"│  {'Left_High':>14s}  {fis.rule_constants[6]:>9.4f}  {fis.rule_constants[7]:>9.4f}  {fis.rule_constants[8]:>9.4f}")
        print(f"└{'─' * 68}")

    print()


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION — Membership Functions
# ═══════════════════════════════════════════════════════════════════════════

def _compute_mfs(x, center):
    """Compute Ruspini-partition MFs: Low, Med, High."""
    c = np.clip(center, 0.001, 0.999)
    low  = np.maximum(0.0, 1.0 - x / c)
    high = np.maximum(0.0, 1.0 - (1.0 - x) / (1.0 - c))
    med  = np.maximum(0.0, 1.0 - low - high)
    return low, med, high


def _bfs_fis_nodes(root):
    """Return FIS nodes in BFS (top-down) order."""
    ordered = []
    queue = [root]
    while queue:
        n = queue.pop(0)
        if isinstance(n, FISNode):
            ordered.append(n)
            if n.left: queue.append(n.left)
            if n.right: queue.append(n.right)
    return ordered


def plot_membership_functions(root):
    """Plot the MFs for every FIS node, showing both input dimensions."""
    fis_nodes_ordered = _bfs_fis_nodes(root)
    n_fis = len(fis_nodes_ordered)
    fig, axes = plt.subplots(n_fis, 2, figsize=(14, 3.5 * n_fis), squeeze=False)
    fig.suptitle("Membership Functions for Each FIS Node", fontsize=16, fontweight='bold', y=1.01)

    x = np.linspace(0, 1, 300)
    colors = ['#2196F3', '#4CAF50', '#FF5722']  # Low=blue, Med=green, High=red

    for row, fis in enumerate(fis_nodes_ordered):
        label = getattr(fis, '_label', None) or _node_label(fis)
        left_name = _node_label(fis.left)
        right_name = _node_label(fis.right)

        for col, (center, input_name) in enumerate([
            (fis.medium1_center, f"Left: {left_name}"),
            (fis.medium2_center, f"Right: {right_name}"),
        ]):
            ax = axes[row, col]
            low, med, high = _compute_mfs(x, center)

            ax.fill_between(x, low,  alpha=0.15, color=colors[0])
            ax.fill_between(x, med,  alpha=0.15, color=colors[1])
            ax.fill_between(x, high, alpha=0.15, color=colors[2])
            ax.plot(x, low,  color=colors[0], lw=2, label='Low')
            ax.plot(x, med,  color=colors[1], lw=2, label='Medium')
            ax.plot(x, high, color=colors[2], lw=2, label='High')

            ax.axvline(center, color='gray', ls='--', lw=1, alpha=0.7)
            ax.text(center, 1.05, f'c={center:.3f}', ha='center', fontsize=9,
                    color='gray', transform=ax.get_xaxis_transform())

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Input Value", fontsize=9)
            ax.set_ylabel("Membership Degree", fontsize=9)
            ax.set_title(f"{input_name}", fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[row, 0].annotate(
            (label or '').replace('\n', ' — '),
            xy=(0, 0.5), xycoords='axes fraction',
            xytext=(-80, 0), textcoords='offset points',
            fontsize=10, fontweight='bold', rotation=90,
            ha='center', va='center', color='#333'
        )

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION — Response Surfaces
# ═══════════════════════════════════════════════════════════════════════════

def _fis_output_surface(center1, center2, rules, resolution=80):
    """Compute the 2D output surface for a single FIS node."""
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    low_a, med_a, high_a = _compute_mfs(X, center1)
    low_b, med_b, high_b = _compute_mfs(Y, center2)

    for i, (ma,) in enumerate([(low_a,), (med_a,), (high_a,)]):
        for j, (mb,) in enumerate([(low_b,), (med_b,), (high_b,)]):
            Z += ma * mb * rules[i * 3 + j]

    return X, Y, Z


def plot_response_surfaces(root):
    """Plot 3D surface + 2D heatmap for every FIS node."""
    fis_nodes_ordered = _bfs_fis_nodes(root)
    n_fis = len(fis_nodes_ordered)
    fig = plt.figure(figsize=(16, 5.5 * n_fis))
    fig.suptitle("FIS Response Surfaces", fontsize=16, fontweight='bold', y=1.0)

    for row, fis in enumerate(fis_nodes_ordered):
        label = getattr(fis, '_label', None) or _node_label(fis)
        left_name = _node_label(fis.left)
        right_name = _node_label(fis.right)

        X, Y, Z = _fis_output_surface(
            fis.medium1_center, fis.medium2_center, fis.rule_constants
        )

        # 3D surface
        ax3d = fig.add_subplot(n_fis, 2, row * 2 + 1, projection='3d')
        ax3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.85, edgecolor='none')
        ax3d.set_xlabel(f"Left: {left_name}", fontsize=8, labelpad=8)
        ax3d.set_ylabel(f"Right: {right_name}", fontsize=8, labelpad=8)
        ax3d.set_zlabel("Output", fontsize=8)
        ax3d.set_title(label, fontsize=10, fontweight='bold', pad=10)
        ax3d.view_init(elev=30, azim=-135)
        ax3d.set_zlim(np.min(Z) - 0.05, np.max(Z) + 0.05)

        # 2D heatmap
        ax2d = fig.add_subplot(n_fis, 2, row * 2 + 2)
        im = ax2d.imshow(
            Z, origin='lower', extent=[0, 1, 0, 1],
            aspect='auto', cmap=cm.viridis, vmin=np.min(Z), vmax=np.max(Z)
        )
        cbar = plt.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
        cbar.set_label("Output", fontsize=9)
        ax2d.set_xlabel(f"Left: {left_name}", fontsize=9)
        ax2d.set_ylabel(f"Right: {right_name}", fontsize=9)
        ax2d.set_title(f"{label} — Heatmap", fontsize=10, fontweight='bold')

        # Annotate the 9 rule constants on the heatmap
        centers_x = [0.0, fis.medium1_center, 1.0]
        centers_y = [0.0, fis.medium2_center, 1.0]
        for i, cx in enumerate(centers_x):
            for j, cy in enumerate(centers_y):
                val = fis.rule_constants[i * 3 + j]
                ax2d.plot(cx, cy, 'wo', markersize=14, markeredgecolor='black', markeredgewidth=0.8)
                ax2d.text(cx, cy, f'{val:.2f}', ha='center', va='center',
                         fontsize=7, fontweight='bold', color='black')

        ax2d.grid(True, alpha=0.2, color='white')

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION — Tree Structure Diagram
# ═══════════════════════════════════════════════════════════════════════════

def plot_tree_structure(root, fontsize=8):
    """Draw the tree as a hierarchical diagram using matplotlib.

    Parameters
    ----------
    root     : FISNode — root of the chromosome tree
    fontsize : float   — base font size for node labels.  Title, legend,
               and other elements scale relative to this value.
    """
    positions = {}
    labels = {}
    colors = {}
    node_id = [0]

    def assign(node, x, y, dx):
        nid = node_id[0]
        node_id[0] += 1
        positions[nid] = (x, y)

        if isinstance(node, InputNode):
            name = INPUT_NAMES.get(node.index, f"In {node.index}")
            labels[nid] = f"[{node.index}]\n{name}"
            colors[nid] = '#FFCDD2'  # light red
            return nid
        else:
            short_label = getattr(node, '_label', None)
            if short_label:
                short_label = short_label.split('\n')[0]
            else:
                short_label = _node_label(node)
            labels[nid] = f"{short_label}\nc1={node.medium1_center:.3f}\nc2={node.medium2_center:.3f}"
            colors[nid] = '#C8E6C9'  # light green
            left_id = assign(node.left, x - dx, y - 1, dx / 2)
            right_id = assign(node.right, x + dx, y - 1, dx / 2)
            return nid, left_id, right_id

    tree_info = assign(root, 0, 0, 3)

    edges = []

    def collect_edges(info):
        if isinstance(info, tuple) and len(info) == 3:
            parent, left, right = info
            left_id = left if isinstance(left, int) else left[0]
            right_id = right if isinstance(right, int) else right[0]
            edges.append((parent, left_id))
            edges.append((parent, right_id))
            collect_edges(left)
            collect_edges(right)

    collect_edges(tree_info)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title("Fuzzy Tree Structure", fontsize=fontsize * 2,
                 fontweight='bold', pad=20)

    for (p, c) in edges:
        px, py = positions[p]
        cx, cy = positions[c]
        ax.plot([px, cx], [py, cy], 'k-', lw=1.5, alpha=0.5, zorder=1)

    for nid, (x, y) in positions.items():
        color = colors[nid]
        bbox = dict(boxstyle="round,pad=0.5", facecolor=color,
                    edgecolor='#555', linewidth=1.5, alpha=0.9)
        ax.text(x, y, labels[nid], ha='center', va='center',
                fontsize=fontsize, fontweight='bold', bbox=bbox, zorder=2)

    ax.set_xlim(min(x for x, y in positions.values()) - 1.5,
                max(x for x, y in positions.values()) + 1.5)
    ax.set_ylim(min(y for x, y in positions.values()) - 1,
                max(y for x, y in positions.values()) + 1)
    ax.set_aspect('equal')
    ax.axis('off')

    legend_fs = fontsize * 1.25
    input_patch = mpatches.Patch(color='#FFCDD2', label='Input Node (leaf)')
    fis_patch = mpatches.Patch(color='#C8E6C9', label='FIS Node (internal)')
    ax.legend(handles=[input_patch, fis_patch], loc='lower right', fontsize=legend_fs)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION — Full-Tree Input Sweeps
# ═══════════════════════════════════════════════════════════════════════════

def plot_input_sweeps(root, resolution=100):
    """
    For each input dimension, sweep it from 0→1 while holding all others
    at their default (0.5), and plot the full tree output.
    """
    compile_chromosome(root)
    n_inputs = 5
    defaults = np.full(n_inputs, 0.5)

    fig, axes = plt.subplots(1, n_inputs, figsize=(4 * n_inputs, 4), sharey=True)
    fig.suptitle("Full-Tree Output: Single-Input Sweeps (others held at 0.5)",
                 fontsize=14, fontweight='bold')

    x = np.linspace(0, 1, resolution)

    for idx in range(n_inputs):
        batch = np.tile(defaults, (resolution, 1))
        batch[:, idx] = x
        outputs = fuzzy_tree_output(root, batch)

        ax = axes[idx]
        ax.plot(x, outputs, lw=2.5, color='#1565C0')
        ax.fill_between(x, outputs, alpha=0.1, color='#1565C0')
        ax.set_xlabel(f"[{idx}] {INPUT_NAMES.get(idx, f'Input {idx}')}", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Tree Output", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Sweep Input {idx}", fontsize=10)

    plt.tight_layout()
    return fig


def plot_pairwise_heatmaps(root, resolution=60):
    """
    Plot pairwise interaction heatmaps: for each pair of inputs,
    sweep both while holding the rest at 0.5.
    """
    compile_chromosome(root)
    n_inputs = 5
    defaults = np.full(n_inputs, 0.5)

    pairs = [(i, j) for i in range(n_inputs) for j in range(i + 1, n_inputs)]
    n_pairs = len(pairs)
    ncols = min(5, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows), squeeze=False)
    fig.suptitle("Full-Tree Output: Pairwise Input Interactions (others at 0.5)",
                 fontsize=14, fontweight='bold')

    x = np.linspace(0, 1, resolution)

    for k, (i, j) in enumerate(pairs):
        row, col = divmod(k, ncols)
        ax = axes[row][col]

        X, Y = np.meshgrid(x, x)
        batch = np.tile(defaults, (resolution * resolution, 1))
        batch[:, i] = X.ravel()
        batch[:, j] = Y.ravel()

        outputs = fuzzy_tree_output(root, batch)
        Z = outputs.reshape(resolution, resolution)

        im = ax.imshow(Z, origin='lower', extent=[0, 1, 0, 1],
                       aspect='auto', cmap=cm.viridis)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(f"[{i}] {INPUT_NAMES.get(i, '')}", fontsize=8)
        ax.set_ylabel(f"[{j}] {INPUT_NAMES.get(j, '')}", fontsize=8)
        ax.set_title(f"In{i} × In{j}", fontsize=9, fontweight='bold')

    for k in range(n_pairs, nrows * ncols):
        row, col = divmod(k, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Inspect and visualize a saved fuzzy tree chromosome (.pkl)")
    parser.add_argument("pkl_file", help="Path to the .pkl chromosome file")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating PNG visualizations (text summary only)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save PNGs but don't open the interactive matplotlib window")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save output PNGs (default: current directory)")
    parser.add_argument("--tree-fontsize", type=float, default=8,
                        help="Base font size for the tree structure diagram (default: 8)")
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  FUZZY TREE CHROMOSOME INSPECTOR")
    print("═" * 70 + "\n")

    # 1. Load
    root = load_chromosome(args.pkl_file)

    # 2. Validate
    validate_tree(root, expected_inputs=5)

    # 3. Clamp and compile
    clamp_params(root, -1.0, 1.0)
    compile_chromosome(root)

    # 4. Print summary
    print_tree_summary(root)

    # 5. Quick test evaluation
    test_input = np.array([[0.8, 0.7, 0.9, 0.2, 0.6]])
    compile_chromosome(root)
    result = fuzzy_tree_output(root, test_input)
    print(f"🧪 Test evaluation:")
    print(f"   Input: Heading=0.8, Closure=0.7, Radius=0.9, Distance=0.2, Collision=0.6")
    print(f"   Output (priority score): {result:.4f}")
    print(f"   (Higher = higher priority target)\n")

    if args.no_plots:
        print("Skipping plots (--no-plots).\n")
        return

    # 6. Generate all plots
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.pkl_file))[0]

    print("📊 Generating visualizations...")

    fig1 = plot_tree_structure(root, fontsize=args.tree_fontsize)
    p1 = os.path.join(args.output_dir, f"{base}_tree_structure.png")
    fig1.savefig(p1, dpi=150, bbox_inches='tight')
    print(f"   ✓ Tree structure → {p1}")

    fig2 = plot_membership_functions(root)
    p2 = os.path.join(args.output_dir, f"{base}_membership_functions.png")
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    print(f"   ✓ Membership functions → {p2}")

    fig3 = plot_response_surfaces(root)
    p3 = os.path.join(args.output_dir, f"{base}_response_surfaces.png")
    fig3.savefig(p3, dpi=150, bbox_inches='tight')
    print(f"   ✓ Response surfaces → {p3}")

    fig4 = plot_input_sweeps(root)
    p4 = os.path.join(args.output_dir, f"{base}_input_sweeps.png")
    fig4.savefig(p4, dpi=150, bbox_inches='tight')
    print(f"   ✓ Input sweeps → {p4}")

    fig5 = plot_pairwise_heatmaps(root)
    p5 = os.path.join(args.output_dir, f"{base}_pairwise_heatmaps.png")
    fig5.savefig(p5, dpi=150, bbox_inches='tight')
    print(f"   ✓ Pairwise interactions → {p5}")

    print(f"\n✅ All done! Output files in: {os.path.abspath(args.output_dir)}\n")

    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()