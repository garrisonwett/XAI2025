import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from algorithms import get_ga_config, run_ga, save_chromosome, InputNode, FISNode
from redone_controller import FuzzyController

###############################################################################
# Tree Visualization
###############################################################################

def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = {root: (xcenter, vert_loc)}
    neighbors = list(G.successors(root))
    if len(neighbors) != 0:
        dx = width / len(neighbors) 
        nextx = xcenter - width/2 - dx/2
        for neighbor in neighbors:
            nextx += dx
            pos.update(hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap, 
                                     vert_loc=vert_loc-vert_gap, xcenter=nextx))
    return pos

def build_graph(node, G, parent, counter):
    nid = f"n{counter[0]}"
    counter[0] += 1

    if isinstance(node, InputNode):
        label = f"In: {node.index}"
        color = "#ffcccc" # Reddish
    elif isinstance(node, FISNode):
        label = f"FIS\n[{node.medium1_center:.2f}, {node.medium2_center:.2f}]"
        color = "#ccffcc" # Greenish
    else:
        label = "?"
        color = "white"

    G.add_node(nid, label=label, fillcolor=color)

    if parent is not None:
        G.add_edge(parent, nid)

    if hasattr(node, "left") and node.left is not None:
        build_graph(node.left, G, nid, counter)
    if hasattr(node, "right") and node.right is not None:
        build_graph(node.right, G, nid, counter)

    return G, nid

def visualize_tree(root):
    G = nx.DiGraph()
    counter = [0]
    build_graph(root, G, None, counter)

    root_nodes = [n for n in G.nodes if G.in_degree(n) == 0]
    if not root_nodes:
        print("Error: Could not find tree root for visualization.")
        return
        
    pos = hierarchy_pos(G, root_nodes[0])
    labels = nx.get_node_attributes(G, "label")
    colors = [nx.get_node_attributes(G, "fillcolor").get(n, "white") for n in G.nodes]

    plt.figure(figsize=(14, 8))
    nx.draw(G, pos, node_color=colors, with_labels=False, arrows=True, node_size=2000, edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold")
    plt.title("Final Fuzzy Tree Structure")
    plt.show()

###############################################################################
# Fitness Plot
###############################################################################

def plot_fitness(history):
    generations = len(history)
    x = np.arange(generations)
    y = np.array(history)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Best Fitness", linewidth=2, marker='o', markersize=4)

    if generations > 1:
        m, b = np.polyfit(x, y, 1)
        yfit = m * x + b
        plt.plot(x, yfit, linestyle="--", color="red", alpha=0.7, label="Trend")

    plt.xlabel("Generation")
    plt.ylabel("Fitness Score (Higher is Better)")
    plt.title("Evolutionary Progress")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()

###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    # 1. Configuration
    cfg = get_ga_config()
    
    # 2. Link your controller class
    # The new 'SafeControllerWrapper' in algorithms.py will handle 
    # if your controller only returns 2 values.
    cfg["controller_callback"] = FuzzyController

    # 3. Train
    best, history = run_ga(cfg)

    # 4. Save & Visualize
    save_chromosome(best, "best_kessler_fuzzy.pkl")
    print("\nTraining complete. Saved to 'best_kessler_fuzzy.pkl'")
    
    try:
        plot_fitness(history)
        visualize_tree(best)
    except Exception as e:
        print(f"Visualization skipped: {e}")