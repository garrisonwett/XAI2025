import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from algorithms import get_ga_config, run_ga, save_chromosome
from redone_controller import FuzzyController


###############################################################################
# Tree Visualization Helpers
###############################################################################

def subtree_color(node, groups_sorted, gather_leaves):
    leaves = []
    gather_leaves(node, leaves)
    leaves_sorted = sorted(leaves)
    for gi, g in enumerate(groups_sorted):
        if leaves_sorted == g:
            return gi
    return -1


def hierarchy_pos(G, root):
    def recurse(n, x0, x1, y, dy, pos):
        pos[n] = ((x0 + x1) / 2, y)
        kids = list(G.successors(n))
        if not kids:
            return pos
        step = (x1 - x0) / len(kids)
        nx0 = x0
        for c in kids:
            nx1 = nx0 + step
            recurse(c, nx0, nx1, y - dy, dy, pos)
            nx0 = nx1
        return pos
    return recurse(root, 0, 1, 0, 0.1, {})


def build_graph(node, groups_sorted, G, parent, counter, gather_leaves):
    nid = f"n{counter[0]}"
    counter[0] += 1

    from algorithms import InputNode, FISNode
    if isinstance(node, InputNode):
        label = f"in {node.index}"
    else:
        label = "FIS"

    color_index = subtree_color(node, groups_sorted, gather_leaves)

    G.add_node(nid, label=label, color=color_index)

    if parent is not None:
        G.add_edge(parent, nid)

    if hasattr(node, "left") and node.left is not None:
        build_graph(node.left, groups_sorted, G, nid, counter, gather_leaves)
    if hasattr(node, "right") and node.right is not None:
        build_graph(node.right, groups_sorted, G, nid, counter, gather_leaves)

    return G


def visualize_tree(root, groups, gather_leaves):
    groups_sorted = [sorted(g) for g in groups]
    G = nx.DiGraph()

    counter = [0]
    G = build_graph(root, groups_sorted, G, None, counter, gather_leaves)

    labels = nx.get_node_attributes(G, "label")
    colors = nx.get_node_attributes(G, "color")

    root_nodes = [n for n in G.nodes if G.in_degree(n) == 0]
    pos = hierarchy_pos(G, root_nodes[0])

    palette = [
        "lightgreen", "lightskyblue", "lightcoral",
        "khaki", "plum", "salmon", "tan", "lightpink"
    ]

    node_colors = []
    for n in G.nodes:
        idx = colors[n]
        if idx is None or idx < 0:
            node_colors.append("lightgray")
        else:
            node_colors.append(palette[idx % len(palette)])

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_color=node_colors, with_labels=False, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.gca().invert_yaxis()
    plt.title("Final Fuzzy Tree Structure")
    plt.show()


###############################################################################
# Fitness Plot
###############################################################################

def plot_fitness(history):
    generations = len(history)
    x = np.arange(generations)
    y = np.array(history)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="Best fitness")

    if generations > 1:
        m, b = np.polyfit(x, y, 1)
        yfit = m * x + b
        plt.plot(x, yfit, linestyle="--", label="Line of best fit")

    plt.xlabel("Generation")
    plt.ylabel("Best fitness (lower is better)")
    plt.title("GA Fitness Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    cfg = get_ga_config()

    cfg["controller_callback"] = lambda chrom: FuzzyController(chrom)

    best, history = run_ga(cfg)

    from algorithms import gather_leaves
    visualize_tree(best, cfg["groups"], gather_leaves)

    plot_fitness(history)

    save_chromosome(best, "best_kessler_fuzzy.pkl")
    print("Training complete.")
    print("Saved best chromosome to best_kessler_fuzzy.pkl")
