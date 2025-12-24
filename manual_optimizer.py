import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from algorithms import InputNode, FISNode, compile_chromosome, save_chromosome, gather_leaves
from redone_controller import FuzzyController
from kesslergame import KesslerGame, TrainerEnvironment
from scenarios import scenarios

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
FILENAME = "hand_optimized.pkl"
SCENARIO = "training2" # or 'asteroid_field', etc.
TEST_EPISODES = 1

# Input Mapping (Matches redone_controller.py logic)
INPUTS = {
    "HEADING": 0,   # 0.0 = On target, 1.0 = Facing away
    "CLOSURE": 1,   # Closing speed (0.0 to 1.0)
    "RADIUS": 2,    # Size of asteroid
    "DISTANCE": 3,  # Distance to asteroid
    "COLLISION": 4  # Time to collision (inverted)
}

# -----------------------------------------------------------------------------
# TREE BUILDING HELPERS
# -----------------------------------------------------------------------------

def make_input(name):
    """Creates a leaf node for a specific input."""
    if name not in INPUTS:
        raise ValueError(f"Unknown input: {name}. Available: {list(INPUTS.keys())}")
    return InputNode(INPUTS[name])

def make_fis(left_node, right_node, m1_center, m2_center, rules_matrix):
    """
    Creates an FIS Node (Intermediate Node).
    
    Args:
        left_node: The child node/input connected to the LEFT input of this node.
        right_node: The child node/input connected to the RIGHT input of this node.
        m1_center (float): 0.01 to 0.99. The peak of the Medium Triangle for the Left Input.
        m2_center (float): 0.01 to 0.99. The peak of the Medium Triangle for the Right Input.
        rules_matrix (list of lists): A 3x3 matrix representing the output constants.
                                      Rows = Left Input (Low, Med, High)
                                      Cols = Right Input (Low, Med, High)
    """
    node = FISNode()
    node.left = left_node
    node.right = right_node
    
    # Set MF Centers
    node.medium1_center = float(np.clip(m1_center, 0.01, 0.99))
    node.medium2_center = float(np.clip(m2_center, 0.01, 0.99))
    
    # Flatten the 3x3 rule matrix into the 1D array of 9 constants
    # Expected order: 
    # [L,L], [L,M], [L,H], 
    # [M,L], [M,M], [M,H], 
    # [H,L], [H,M], [H,H]
    flat_rules = []
    if len(rules_matrix) != 3 or any(len(row) != 3 for row in rules_matrix):
        raise ValueError("Rules must be a 3x3 list of lists.")
        
    for row in rules_matrix:
        flat_rules.extend(row)
        
    node.rule_constants = flat_rules
    return node

# -----------------------------------------------------------------------------
# VISUALIZATION (Copied from main_train.py for standalone usage)
# -----------------------------------------------------------------------------
def subtree_color(node, gather_leaves):
    # Simplified coloring based on just getting unique ID of inputs
    leaves = []
    gather_leaves(node, leaves)
    return hash(tuple(sorted(leaves))) % 8

def hierarchy_pos(G, root):
    def recurse(n, x0, x1, y, dy, pos):
        pos[n] = ((x0 + x1) / 2, y)
        kids = list(G.successors(n))
        if not kids: return pos
        step = (x1 - x0) / len(kids)
        nx0 = x0
        for c in kids:
            nx1 = nx0 + step
            recurse(c, nx0, nx1, y - dy, dy, pos)
            nx0 = nx1
        return pos
    return recurse(root, 0, 1, 0, 0.1, {})

def build_graph(node, G, parent, counter):
    nid = f"n{counter[0]}"
    counter[0] += 1
    
    if isinstance(node, InputNode):
        # Reverse lookup name
        name = [k for k, v in INPUTS.items() if v == node.index][0]
        label = f"{name}"
        color = 0
    else:
        label = f"FIS\nM1:{node.medium1_center:.2f}\nM2:{node.medium2_center:.2f}"
        color = 1

    G.add_node(nid, label=label, color=color)
    if parent: G.add_edge(parent, nid)
    
    if hasattr(node, "left") and node.left:
        build_graph(node.left, G, nid, counter)
    if hasattr(node, "right") and node.right:
        build_graph(node.right, G, nid, counter)
    return G

def show_tree(root):
    G = nx.DiGraph()
    build_graph(root, G, None, [0])
    
    pos = hierarchy_pos(G, [n for n in G.nodes if G.in_degree(n)==0][0])
    labels = nx.get_node_attributes(G, 'label')
    
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=False, node_color="lightblue", node_size=1500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title("Hand-Designed Fuzzy Tree")
    plt.show()

# -----------------------------------------------------------------------------
# HAND OPTIMIZATION SECTION (EDIT THIS FUNCTION)
# -----------------------------------------------------------------------------

def build_manual_chromosome():
    """
    Construct your tree here.
    
    Rule Matrix Key:
    -1.0 = Highly Undesirable (Low Threat Score)
     0.0 = Neutral
     1.0 = Highly Desirable (High Threat Score - Target this asteroid!)
    
    Inputs Reminder:
    - HEADING: 0 (Good/Front), 1 (Bad/Behind)
    - DISTANCE: 0 (Close), 1 (Far)
    - COLLISION: 0 (Safe), 1 (Imminent Impact)
    """

    # --- SUB-TREE 1: GEOMETRY (Heading vs Distance) ---
    # Logic: We prefer asteroids that are In Front (Low Heading) and Close (Low Distance).
    # If it is behind us (High Heading), we almost never want to target it unless it's very close.
    
    # Left Input: HEADING (Low=Front, High=Back) -> Center 0.3
    # Right Input: DISTANCE (Low=Close, High=Far) -> Center 0.3
    geometry_node = make_fis(
        left_node  = make_input("HEADING"),
        right_node = make_input("DISTANCE"),
        m1_center  = 0.3, # Heading cutoff
        m2_center  = 0.3, # Distance cutoff
        rules_matrix = [
            # D_Low, D_Med, D_High
            [ 1.0,   0.6,   0.1],  # Head_Low (Front) -> Prefer Close
            [ 0.8,   0.4,  0.1],  # Head_Med (Side)
            [0.6,  0.2,  0]   # Head_High (Behind) -> Ignore
        ]
    )

    # --- SUB-TREE 2: URGENCY (Collision vs Radius) ---
    # Logic: Prioritize Collision threats. If no threat, maybe prefer large asteroids (split easier).
    
    # Left Input: COLLISION (0=Safe, 1=Danger)
    # Right Input: RADIUS (0=Small, 1=Large)
    urgency_node = make_fis(
        left_node  = make_input("COLLISION"),
        right_node = make_input("RADIUS"),
        m1_center  = 0.5,
        m2_center  = 0.5,
        rules_matrix = [
            # Rad_S, Rad_M, Rad_L
            [ 0.0,   0.1,   0.2],  # Coll_Safe -> Slight pref for large
            [ 0.5,   0.6,   0.7],  # Coll_Med
            [ 1.0,   1.0,   1.0]   # Coll_High -> MAX PRIORITY regardless of size
        ]
    )

    # --- ROOT NODE: COMBINE GEOMETRY AND URGENCY ---
    # Logic: If Urgency is High, override Geometry. If Urgency is Low, use Geometry.
    
    root = make_fis(
        left_node  = geometry_node,
        right_node = urgency_node,
        m1_center  = 0.5, # Cutoff for Geometry score
        m2_center  = 0.5, # Cutoff for Urgency score
        rules_matrix = [
            # Urg_L, Urg_M, Urg_H
            [-1.0,   0.0,   1.0], # Geo_Low (Bad Geometry) -> Only target if High Urgency
            [ 0.0,   0.5,   1.0], # Geo_Med
            [ 1.0,   1.0,   1.0]  # Geo_High (Good Geometry) -> Always target unless... wait, Urgency usually wins
        ]
    )
    
    # Override logic: If Urgency (Right) is High, output is High (1.0).
    # If Urgency is Low, we rely on Geometry (Left).
    
    return root

# -----------------------------------------------------------------------------
# EVALUATION LOOP
# -----------------------------------------------------------------------------

def evaluate_manual_tree(root):
    # 1. Compile for Numba
    compile_chromosome(root)
    
    # 2. Setup Game
    print(f"Running {TEST_EPISODES} episodes on scenario: {SCENARIO}")
    game_settings = {
        "frequency": 30,
        "perf_tracker": False,
        "prints_on": False,
        "graphics_type": 0,
        "realtime_multiplier": 0,
        "time_limit": 120.0,
    }
    
    total_score = 0
    scenario_obj = scenarios[SCENARIO]
    
    for i in range(TEST_EPISODES):
        game = TrainerEnvironment(settings=game_settings)
        # Create controller with our specific tree
        controller = FuzzyController(root) 
        score, info = game.run(scenario=scenario_obj, controllers=[controller])
        
        # Calculate score (matches algorithms.py)
        t = score.teams[0]
        s = (t.asteroids_hit * t.accuracy) - 20 * t.deaths
        total_score += s
        
        print(f"  Episode {i+1}: Score = {s:.2f} (Hits: {t.asteroids_hit}, Acc: {t.accuracy*100:.1f}%, Deaths: {t.deaths})")

    avg = total_score / TEST_EPISODES
    print("="*40)
    print(f"AVERAGE FITNESS: {avg:.4f}")
    print("="*40)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Build
    print("Building tree...")
    tree = build_manual_chromosome()
    
    # 2. Visualize
    print("Displaying tree structure...")
    try:
        show_tree(tree)
    except Exception as e:
        print(f"Could not visualize: {e}")
    
    # 3. Save
    save_chromosome(tree, FILENAME)
    print(f"Saved chromosome to {FILENAME}")
    
    # 4. Test
    print("Testing performance...")
    evaluate_manual_tree(tree)