#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║          FUZZY TREE HAND-TUNING & VISUALIZATION TOOL                   ║
║                                                                        ║
║  Create, inspect, and export hand-crafted fuzzy tree chromosomes       ║
║  for use as baselines / comparisons against the GA-evolved agents.     ║
╚══════════════════════════════════════════════════════════════════════════╝

USAGE
─────
  1.  Edit the `define_hand_tuned_tree()` function below to set your
      desired tree topology, MF centers, and rule constants.

  2.  Run:  python hand_tune_chromosome.py

  3.  The script will:
        • Print a text summary of every node
        • Show a tree-structure diagram
        • Plot membership functions for each FIS node
        • Plot the 3-D response surface for each FIS node
        • Plot a combined "inputs sweep" showing the full tree output
        • Save the chromosome to a .pkl file ready for visualize_run.py

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
from turtle import heading
from networkx import radius
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Import the tree node classes from your project.
# Adjust the path if needed.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from algorithms import (
        InputNode, FISNode, compile_chromosome, clamp_params,
        save_chromosome, copy_tree, get_tree_size, fuzzy_tree_output,
        gather_fis_nodes, gather_leaf_nodes, flatten_tree,
    )
except ImportError:
    # Fallback: copy minimal class definitions so the tool works standalone
    print("[WARN] Could not import from algorithms.py — using built-in copies.")
    print("       Place this file next to algorithms.py for full compatibility.\n")

    class InputNode:
        def __init__(self, idx):
            self.index = idx
            self.left = self.right = None

    class FISNode:
        def __init__(self):
            self.medium1_center = 0.5
            self.medium2_center = 0.5
            self.rule_constants = [0.0] * 9
            self.left = self.right = None


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1:  DEFINE YOUR HAND-TUNED TREE HERE
# ═══════════════════════════════════════════════════════════════════════════

# Friendly names for each input index (used in plots / printouts)
INPUT_NAMES = {
    0: "Heading Alignment",
    1: "Closure Rate",
    2: "Asteroid Radius",
    3: "Log Distance",
    4: "Collision Urgency",
}


def make_fis(left_child, right_child, center1, center2, rules, label=""):
    """
    Convenience builder for a single FIS node.

    Parameters
    ──────────
    left_child   : InputNode or FISNode — the left (first) input source
    right_child  : InputNode or FISNode — the right (second) input source
    center1      : float in (0, 1) — Medium-MF center for the LEFT input
    center2      : float in (0, 1) — Medium-MF center for the RIGHT input
    rules        : list of 9 floats — rule consequent constants
    label        : str — human-readable name (for plot titles)
    """
    node = FISNode()
    node.medium1_center = center1
    node.medium2_center = center2
    node.rule_constants = list(rules)
    node.left = left_child
    node.right = right_child
    node._label = label  # extra attribute for visualization only
    return node


def define_hand_tuned_tree():
    """
    ╔════════════════════════════════════════════════════════════════════╗
    ║  EDIT THIS FUNCTION TO DEFINE YOUR HAND-TUNED CHROMOSOME.       ║
    ║                                                                  ║
    ║  The tree must use each input index (0–4) exactly once as a     ║
    ║  leaf, and every internal node must be a FISNode with exactly   ║
    ║  two children.                                                   ║
    ╚════════════════════════════════════════════════════════════════════╝

    Current topology (edit to taste):

                         root (FIS_C)
                        /            \\
                   FIS_A              FIS_B
                  /     \\            /     \\
          Heading(0)  Distance(3)  Radius(2)  Collision(4)
                          \\
                       Closure(1) ← wait, that's wrong...

    Actually let's use a clean 5-input tree:

                         root (FIS_D)
                        /            \\
                   FIS_A              FIS_C
                  /     \\            /     \\
          Heading(0)  Closure(1)  FIS_B    Collision(4)
                                 /    \\
                          Radius(2)  Distance(3)

    Intuition behind this grouping:
      • FIS_B combines Radius + Distance → "how threatening is this asteroid
        purely by geometry" (big + close = high threat).
      • FIS_C combines that geometric threat with Collision Urgency → "how
        urgent is the overall danger from this asteroid".
      • FIS_A combines Heading + Closure → "how easy/rewarding is this target
        to engage" (ahead of us + approaching = great opportunity).
      • Root (FIS_D) balances engagement opportunity vs. danger urgency →
        final priority score.
    """

    # ── Leaves ──────────────────────────────────────────────────────────
    heading   = InputNode(0)   # Heading Alignment
    closure   = InputNode(1)   # Closure Rate
    radius    = InputNode(2)   # Asteroid Radius
    distance  = InputNode(3)   # Log Distance
    collision = InputNode(4)   # Collision Urgency

    # ── FIS_B: Radius × Distance → Geometric Threat ────────────────────
    #
    #  We want HIGH output when the asteroid is BIG (radius→1) and
    #  CLOSE (distance→0, i.e. Low distance).
    #
    #  Rule table (Left=Radius, Right=Distance):
    #              Dist_Low(close) Dist_Med  Dist_High(far)
    #  Rad_Low      0.1            0.0        0.0       ← small asteroid, ignore
    #  Rad_Med      0.6            0.3        0.1       ← medium: prioritize if close
    #  Rad_High     1.0            0.7        0.3       ← big: always somewhat important
    #

    # ── FIS_C: Geometric Threat × Collision Urgency → Danger ───────────
    #
    #  HIGH output when geometric threat is high AND collision is imminent.
    #
    #              Coll_Low(safe) Coll_Med  Coll_High(imminent)
    #  Geo_Low      0.0           0.1        0.3      ← low geo threat, some bump if colliding
    #  Geo_Med      0.2           0.4        0.7      ← moderate threat
    #  Geo_High     0.4           0.7        1.0      ← high threat + imminent = max danger
    #

    # ── FIS_A: Heading × Closure → Engagement Opportunity ──────────────
    #
    #  HIGH output when asteroid is AHEAD (heading→1) and APPROACHING (closure→1).
    #
    #              Clos_Low(sep)  Clos_Med  Clos_High(approach)
    #  Head_Low     0.0           0.1        0.2      ← behind us, hard to engage
    #  Head_Med     0.2           0.4        0.6      ← somewhat aligned
    #  Head_High    0.3           0.6        1.0      ← ahead + approaching = best target
    #


    # ── Root (FIS_D): Engagement × Danger → Final Priority ─────────────
    #
    #  We want to balance "can I easily shoot this" vs "do I need to deal
    #  with this urgently".  Danger gets a slight edge because survival matters.
    #
    #              Danger_Low     Danger_Med  Danger_High
    #  Engage_Low   0.0           0.3         0.5     ← can't engage but dangerous → still some prio
    #  Engage_Med   0.3           0.5         0.8     ← moderate on both
    #  Engage_High  0.5           0.7         1.0     ← easy to engage + dangerous = top priority
    #


    fis_a = make_fis(
        left_child  = heading,
        right_child = radius,
        center1     = 0.80,
        center2     = 0.40,   
        rules       = [0.3, 0.1, 0.0,
                       0.7, 0.4, 0.2,
                       1.0, 0.7, 0.4],
        label       = "FIS_A: Angular Opportunity\n(Heading × Radius)",
    )



    fis_b = make_fis(
        left_child  = closure,
        right_child = collision,
        center1     = 0.40,   
        center2     = 0.35,   
        rules       = [1.0, 0.8, 0.4,
                       0.6, 0.3, 0.1,
                       0.4, 0.2, 0.0],
        label       = "FIS_B: Geometric Threat\n(Closure × Collision)",
    )



    fis_c = make_fis(
        left_child  = fis_b,
        right_child = distance,
        center1     = 0.50,
        center2     = 0.50,
        rules       = [0.3, 0.1, 0.0,
                       0.6, 0.4, 0.2,
                       1.0, 0.6, 0.3],
        label       = "FIS_C: Location Danger\n(Geometric Threat × Closure)",
    )

    root = make_fis(
        left_child  = fis_a,
        right_child = fis_c,
        center1     = 0.50,
        center2     = 0.50,
        rules       = [0.0, 0.3, 0.5,
                       0.3, 0.5, 0.8,
                       0.5, 0.7, 1.0],
        label       = "FIS_D: Final Priority\n(Engagement Opportunity × Location Danger)",
    )

    return root


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2:  VALIDATION
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
        return getattr(node, '_label', 'FIS').split('\n')[0]
    return "?"


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3:  TEXT SUMMARY
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
        label = getattr(fis, '_label', f'FIS Node {i}')
        left_label = _node_label(fis.left)
        right_label = _node_label(fis.right)

        print(f"\n┌─ {label}")
        print(f"│  Left input:  {left_label}   (MF center = {fis.medium1_center:.2f})")
        print(f"│  Right input: {right_label}   (MF center = {fis.medium2_center:.2f})")
        print(f"│")
        print(f"│  Rule Table:")
        print(f"│  {'':>14s}  {'Right_Low':>9s}  {'Right_Med':>9s}  {'Right_Hi':>9s}")
        print(f"│  {'Left_Low':>14s}  {fis.rule_constants[0]:>9.3f}  {fis.rule_constants[1]:>9.3f}  {fis.rule_constants[2]:>9.3f}")
        print(f"│  {'Left_Med':>14s}  {fis.rule_constants[3]:>9.3f}  {fis.rule_constants[4]:>9.3f}  {fis.rule_constants[5]:>9.3f}")
        print(f"│  {'Left_High':>14s}  {fis.rule_constants[6]:>9.3f}  {fis.rule_constants[7]:>9.3f}  {fis.rule_constants[8]:>9.3f}")
        print(f"└{'─' * 68}")

    print()


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4:  VISUALIZATION — Membership Functions
# ═══════════════════════════════════════════════════════════════════════════

def _compute_mfs(x, center):
    """Compute Ruspini-partition MFs: Low, Med, High."""
    c = np.clip(center, 0.001, 0.999)
    low  = np.maximum(0.0, 1.0 - x / c)
    high = np.maximum(0.0, 1.0 - (1.0 - x) / (1.0 - c))
    med  = np.maximum(0.0, 1.0 - low - high)
    return low, med, high


def plot_membership_functions(root):
    """Plot the MFs for every FIS node, showing both input dimensions."""
    fis_nodes = list(_iter_fis(root))
    # Reverse to get top-down BFS order
    fis_nodes_ordered = []
    queue = [root]
    while queue:
        n = queue.pop(0)
        if isinstance(n, FISNode):
            fis_nodes_ordered.append(n)
            if n.left: queue.append(n.left)
            if n.right: queue.append(n.right)

    n_fis = len(fis_nodes_ordered)
    fig, axes = plt.subplots(n_fis, 2, figsize=(14, 3.5 * n_fis), squeeze=False)
    fig.suptitle("Membership Functions for Each FIS Node", fontsize=16, fontweight='bold', y=1.01)

    x = np.linspace(0, 1, 300)
    colors = ['#2196F3', '#4CAF50', '#FF5722']  # Low=blue, Med=green, High=red

    for row, fis in enumerate(fis_nodes_ordered):
        label = getattr(fis, '_label', f'FIS {row}')
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
            ax.text(center, 1.05, f'c={center:.2f}', ha='center', fontsize=9,
                    color='gray', transform=ax.get_xaxis_transform())

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Input Value", fontsize=9)
            ax.set_ylabel("Membership Degree", fontsize=9)
            ax.set_title(f"{input_name}", fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Add the FIS label on the left margin
        axes[row, 0].annotate(
            label.replace('\n', ' — '),
            xy=(0, 0.5), xycoords='axes fraction',
            xytext=(-80, 0), textcoords='offset points',
            fontsize=10, fontweight='bold', rotation=90,
            ha='center', va='center', color='#333'
        )

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5:  VISUALIZATION — Response Surfaces
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
    fis_nodes_ordered = []
    queue = [root]
    while queue:
        n = queue.pop(0)
        if isinstance(n, FISNode):
            fis_nodes_ordered.append(n)
            if n.left: queue.append(n.left)
            if n.right: queue.append(n.right)

    n_fis = len(fis_nodes_ordered)
    fig = plt.figure(figsize=(16, 5.5 * n_fis))
    fig.suptitle("FIS Response Surfaces", fontsize=16, fontweight='bold', y=1.0)

    for row, fis in enumerate(fis_nodes_ordered):
        label = getattr(fis, '_label', f'FIS {row}')
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
        mf_labels_x = ['L', 'M', 'H']
        mf_labels_y = ['L', 'M', 'H']
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
#  STEP 6:  VISUALIZATION — Tree Structure Diagram
# ═══════════════════════════════════════════════════════════════════════════

def plot_tree_structure(root):
    """Draw the tree as a hierarchical diagram using matplotlib (no networkx needed)."""
    # Assign positions via recursive layout
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
            short_label = getattr(node, '_label', 'FIS').split('\n')[0]
            labels[nid] = f"{short_label}\nc1={node.medium1_center:.2f}\nc2={node.medium2_center:.2f}"
            colors[nid] = '#C8E6C9'  # light green
            left_id = assign(node.left, x - dx, y - 1, dx / 2)
            right_id = assign(node.right, x + dx, y - 1, dx / 2)
            return nid, left_id, right_id

    # Build structure
    tree_info = assign(root, 0, 0, 3)

    # Collect edges
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

    # Plot
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title("Fuzzy Tree Structure", fontsize=16, fontweight='bold', pad=20)

    # Draw edges
    for (p, c) in edges:
        px, py = positions[p]
        cx, cy = positions[c]
        ax.plot([px, cx], [py, cy], 'k-', lw=1.5, alpha=0.5, zorder=1)

    # Draw nodes
    for nid, (x, y) in positions.items():
        color = colors[nid]
        bbox = dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor='#555',
                    linewidth=1.5, alpha=0.9)
        ax.text(x, y, labels[nid], ha='center', va='center',
                fontsize=8, fontweight='bold', bbox=bbox, zorder=2)

    ax.set_xlim(min(x for x, y in positions.values()) - 1.5,
                max(x for x, y in positions.values()) + 1.5)
    ax.set_ylim(min(y for x, y in positions.values()) - 1,
                max(y for x, y in positions.values()) + 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend
    input_patch = mpatches.Patch(color='#FFCDD2', label='Input Node (leaf)')
    fis_patch = mpatches.Patch(color='#C8E6C9', label='FIS Node (internal)')
    ax.legend(handles=[input_patch, fis_patch], loc='lower right', fontsize=10)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 7:  VISUALIZATION — Full-Tree Input Sweeps
# ═══════════════════════════════════════════════════════════════════════════

def plot_input_sweeps(root, resolution=100):
    """
    For each input dimension, sweep it from 0→1 while holding all others
    at their default (0.5), and plot the full tree output.

    This shows the marginal sensitivity of the tree to each input.
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

    # Hide unused subplots
    for k in range(n_pairs, nrows * ncols):
        row, col = divmod(k, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 70)
    print("  FUZZY TREE HAND-TUNING TOOL")
    print("═" * 70 + "\n")

    # 1. Build the tree
    root = define_hand_tuned_tree()

    # 2. Validate
    validate_tree(root, expected_inputs=5)

    # 3. Clamp and compile
    clamp_params(root, -1.0, 1.0)
    compile_chromosome(root)

    # 4. Print summary
    print_tree_summary(root)

    # 5. Save chromosome
    output_filename = "hand_tuned_chromosome.pkl"
    save_chromosome(root, output_filename)
    print(f"💾 Chromosome saved to: {output_filename}")
    print(f"   Use with:  python visualize_run.py --chromosome_file {output_filename}\n")

    # 6. Quick test evaluation
    test_input = np.array([[0.8, 0.7, 0.9, 0.2, 0.6]])  # ahead, approaching, big, close, moderate urgency
    compile_chromosome(root)
    result = fuzzy_tree_output(root, test_input)
    print(f"🧪 Test evaluation:")
    print(f"   Input: Heading=0.8, Closure=0.7, Radius=0.9, Distance=0.2, Collision=0.6")
    print(f"   Output (priority score): {result:.4f}")
    print(f"   (Higher = higher priority target)\n")

    # 7. Generate all plots
    print("📊 Generating visualizations...")

    fig1 = plot_tree_structure(root)
    fig1.savefig("hand_tune_tree_structure.png", dpi=150, bbox_inches='tight')
    print("   ✓ Tree structure → hand_tune_tree_structure.png")

    fig2 = plot_membership_functions(root)
    fig2.savefig("hand_tune_membership_functions.png", dpi=150, bbox_inches='tight')
    print("   ✓ Membership functions → hand_tune_membership_functions.png")

    fig3 = plot_response_surfaces(root)
    fig3.savefig("hand_tune_response_surfaces.png", dpi=150, bbox_inches='tight')
    print("   ✓ Response surfaces → hand_tune_response_surfaces.png")

    fig4 = plot_input_sweeps(root)
    fig4.savefig("hand_tune_input_sweeps.png", dpi=150, bbox_inches='tight')
    print("   ✓ Input sweeps → hand_tune_input_sweeps.png")

    fig5 = plot_pairwise_heatmaps(root)
    fig5.savefig("hand_tune_pairwise_heatmaps.png", dpi=150, bbox_inches='tight')
    print("   ✓ Pairwise interactions → hand_tune_pairwise_heatmaps.png")

    print("\n✅ All done! Open the PNG files to inspect your design.")
    print("   Edit define_hand_tuned_tree() and re-run to iterate.\n")

    # Show plots interactively if possible
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()