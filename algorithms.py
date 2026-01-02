import numpy as np
import random
import pickle
import copy
import time
import os
from concurrent.futures import ProcessPoolExecutor
from numba import njit, float64, int32
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------------------------------------------------------
# SECTION 1: CONFIGURATION
# -----------------------------------------------------------------------------

def get_ga_config():
    cfg = {
        # --- Genetic Algorithm Settings ---
        "popsize": 40,                
        "generations": 250,
        "tournament_k": 3,
        
        # --- Multiprocessing ---
        "num_workers": 8,             # Adjust based on your CPU cores
        
        # --- Problem Constraints ---
        "input_count": 5,             
        "groups": [],                 
        "max_hours": 2.0,             

        # --- Mutation Rates ---
        "mf_mut_rate_start": 0.50, "mf_mut_rate_end": 0.02,
        "rule_mut_rate_start": 0.50, "rule_mut_rate_end": 0.02,
        "struct_mut_prob_start": 0.60, "struct_mut_prob_end": 0.10,
        "param_cross_prob_start": 0.80, "param_cross_prob_end": 0.50,

        # --- Evaluation ---
        "scenario_name": "training2", 
        "episodes_per_eval": 1,
        "controller_callback": None, 
    }
    return cfg

def linear_schedule(start, end, gen, total):
    if total <= 1: return end
    return start + (end - start) * (gen / (total - 1))

# -----------------------------------------------------------------------------
# SECTION 2: FUZZY TREE NODE DEFINITIONS
# -----------------------------------------------------------------------------

class InputNode:
    def __init__(self, idx):
        self.index = idx
        self.left = None
        self.right = None
    def __repr__(self): return f"InputNode({self.index})"

class FISNode:
    def __init__(self):
        self.medium1_center = random.uniform(0.2, 0.8)
        self.medium2_center = random.uniform(0.2, 0.8)
        self.rule_constants = [random.uniform(-1, 1) for _ in range(9)]
        self.left = None
        self.right = None
    def __repr__(self): return f"FISNode(c1={self.medium1_center:.2f})"

# -----------------------------------------------------------------------------
# SECTION 3: TREE UTILITIES
# -----------------------------------------------------------------------------

def get_tree_size(node):
    count = 0
    stack = [node]
    while stack:
        n = stack.pop()
        count += 1
        if isinstance(n, FISNode):
            if n.left: stack.append(n.left)
            if n.right: stack.append(n.right)
    return count

def gather_leaves(node, lst):
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, InputNode): lst.append(n.index)
        else:
            if n.right: stack.append(n.right)
            if n.left: stack.append(n.left)

def gather_leaf_nodes(node, lst):
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, InputNode): lst.append(n)
        else:
            if n.right: stack.append(n.right)
            if n.left: stack.append(n.left)

def gather_fis_nodes(node, lst):
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, FISNode):
            lst.append(n)
            if n.right: stack.append(n.right)
            if n.left: stack.append(n.left)

def clamp_mfs(node):
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, FISNode):
            n.medium1_center = min(max(n.medium1_center, 0.01), 0.99)
            n.medium2_center = min(max(n.medium2_center, 0.01), 0.99)
            if n.left: stack.append(n.left)
            if n.right: stack.append(n.right)

def copy_tree(node):
    if isinstance(node, InputNode): return InputNode(node.index)
    root_copy = FISNode()
    root_copy.medium1_center = node.medium1_center
    root_copy.medium2_center = node.medium2_center
    root_copy.rule_constants = node.rule_constants[:]
    stack = [(node, root_copy)]
    while stack:
        orig, new = stack.pop()
        if orig.left:
            if isinstance(orig.left, InputNode): new.left = InputNode(orig.left.index)
            else:
                child = FISNode()
                child.medium1_center = orig.left.medium1_center
                child.medium2_center = orig.left.medium2_center
                child.rule_constants = orig.left.rule_constants[:]
                new.left = child
                stack.append((orig.left, child))
        if orig.right:
            if isinstance(orig.right, InputNode): new.right = InputNode(orig.right.index)
            else:
                child = FISNode()
                child.medium1_center = orig.right.medium1_center
                child.medium2_center = orig.right.medium2_center
                child.rule_constants = orig.right.rule_constants[:]
                new.right = child
                stack.append((orig.right, child))
    return root_copy

# -----------------------------------------------------------------------------
# SECTION 4: INITIALIZATION
# -----------------------------------------------------------------------------

def random_full_tree_with_leaves(indices):
    leaves = [InputNode(i) for i in indices]
    random.shuffle(leaves)
    nodes = leaves[:]
    while len(nodes) > 1:
        a = nodes.pop(random.randrange(len(nodes)))
        b = nodes.pop(random.randrange(len(nodes)))
        p = FISNode()
        p.left, p.right = a, b
        nodes.append(p)
    root = nodes[0]
    clamp_mfs(root)
    return root

def build_initial_tree(count, groups_sorted):
    all_inputs = list(range(count))
    used = set()
    subs = []
    for g in groups_sorted:
        valid_g = [x for x in g if x < count and x not in used]
        if valid_g:
            subs.append(random_full_tree_with_leaves(valid_g))
            for x in valid_g: used.add(x)
    free = [i for i in all_inputs if i not in used]
    if free: subs.append(random_full_tree_with_leaves(free))
    nodes = subs[:]
    if not nodes: return random_full_tree_with_leaves(all_inputs)
    while len(nodes) > 1:
        a = nodes.pop(random.randrange(len(nodes)))
        b = nodes.pop(random.randrange(len(nodes)))
        p = FISNode()
        p.left, p.right = a, b
        nodes.append(p)
    root = nodes[0]
    clamp_mfs(root)
    return root

# -----------------------------------------------------------------------------
# SECTION 5: COMPILATION
# -----------------------------------------------------------------------------

def flatten_tree(root):
    order = []
    stack = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited: order.append(node)
        else:
            stack.append((node, True))
            if isinstance(node, FISNode):
                if node.right: stack.append((node.right, False))
                if node.left: stack.append((node.left, False))

    index_map = {node: i for i, node in enumerate(order)}
    n = len(order)
    node_type = np.zeros(n, dtype=np.int32)
    left = np.zeros(n, dtype=np.int32)
    right = np.zeros(n, dtype=np.int32)
    m1_inv, m1_inv_c = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    m2_inv, m2_inv_c = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    rules = np.zeros((n, 9), dtype=np.float64)

    for i, node in enumerate(order):
        if isinstance(node, InputNode):
            node_type[i], left[i], right[i] = 0, node.index, node.index
        else:
            node_type[i] = 1
            left[i], right[i] = index_map[node.left], index_map[node.right]
            c1, c2 = node.medium1_center, node.medium2_center
            c1 = min(max(c1, 0.001), 0.999)
            c2 = min(max(c2, 0.001), 0.999)
            m1_inv[i], m1_inv_c[i] = 1.0/c1, 1.0/(1.0-c1)
            m2_inv[i], m2_inv_c[i] = 1.0/c2, 1.0/(1.0-c2)
            rules[i, :] = node.rule_constants
    return node_type, left, right, m1_inv, m1_inv_c, m2_inv, m2_inv_c, rules

@njit(fastmath=True)
def fis_eval_batch(node_type, left, right, m1_inv, m1_inv_c, m2_inv, m2_inv_c, rules, inputs_batch):
    num_samples = inputs_batch.shape[0]
    num_nodes = node_type.shape[0]
    total_size = num_samples * num_nodes
    node_outputs = np.empty(total_size, dtype=np.float64)
    
    for k in range(num_samples):
        k_offset = k * num_nodes
        for i in range(num_nodes):
            if node_type[i] == 0:
                node_outputs[k_offset + i] = inputs_batch[k, left[i]]
            else:
                val_a = node_outputs[k_offset + left[i]]
                val_b = node_outputs[k_offset + right[i]]
                
                low_a = max(0.0, 1.0 - (val_a * m1_inv[i]))
                high_a = max(0.0, 1.0 - ((1.0 - val_a) * m1_inv_c[i]))
                med_a = max(0.0, 1.0 - low_a - high_a)

                low_b = max(0.0, 1.0 - (val_b * m2_inv[i]))
                high_b = max(0.0, 1.0 - ((1.0 - val_b) * m2_inv_c[i]))
                med_b = max(0.0, 1.0 - low_b - high_b)

                r_low_a = (low_b * rules[i, 0]) + (med_b * rules[i, 1]) + (high_b * rules[i, 2])
                r_med_a = (low_b * rules[i, 3]) + (med_b * rules[i, 4]) + (high_b * rules[i, 5])
                r_high_a = (low_b * rules[i, 6]) + (med_b * rules[i, 7]) + (high_b * rules[i, 8])
                
                node_outputs[k_offset + i] = (low_a * r_low_a) + (med_a * r_med_a) + (high_a * r_high_a)

    final_output = np.empty(num_samples, dtype=np.float64)
    for k in range(num_samples):
        final_output[k] = node_outputs[k * num_nodes + (num_nodes - 1)]
    return final_output

def compile_chromosome(chrom):
    # CRITICAL: If this is a Dummy object from a worker process, it is already flat.
    # Do not attempt to flatten it again.
    if hasattr(chrom, "is_dummy"):
        return
        
    chrom._flat_repr = flatten_tree(chrom)

def fuzzy_tree_output(chrom, *inputs):
    if not hasattr(chrom, "_flat_repr"): compile_chromosome(chrom)
    nt, le, ri, m1i, m1ic, m2i, m2ic, rl = chrom._flat_repr
    if len(inputs) == 1 and isinstance(inputs[0], np.ndarray):
        arr = inputs[0]
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        res = fis_eval_batch(nt, le, ri, m1i, m1ic, m2i, m2ic, rl, arr)
        return res[0] if res.shape[0] == 1 else res
    else:
        arr = np.array([inputs], dtype=np.float64) 
        res = fis_eval_batch(nt, le, ri, m1i, m1ic, m2i, m2ic, rl, arr)
        return res[0]

def save_chromosome(ch, filename):
    if hasattr(ch, "compiled"): del ch.compiled
    if hasattr(ch, "_flat_repr"): del ch._flat_repr
    if hasattr(ch, "cached_fitness"): del ch.cached_fitness
    with open(filename, "wb") as f: pickle.dump(ch, f)
    compile_chromosome(ch)

def load_chromosome(filename):
    with open(filename, "rb") as f: ch = pickle.load(f)
    compile_chromosome(ch)
    return ch

# -----------------------------------------------------------------------------
# SECTION 6: WORKER EVALUATION & SAFETY WRAPPER
# -----------------------------------------------------------------------------

try:
    from kesslergame import KesslerGame, TrainerEnvironment
    from scenarios import scenarios
except ImportError:
    print("Warning: kesslergame not found.")

game_settings = {
    "frequency": 30,
    "perf_tracker": False,
    "prints_on": False,
    "graphics_type": 0,
    "realtime_multiplier": 0,
    "time_limit": 120.0,
}

def kessler_score_to_scalar(score, info):
    t = score.teams[0]
    return (t.asteroids_hit * t.accuracy) - 20 * t.deaths 

# --- SAFETY WRAPPER ---
class SafeControllerWrapper:
    def __init__(self, controller):
        self.controller = controller
        
    @property
    def name(self):
        return self.controller.name
        
    def actions(self, ship_state, game_state):
        result = self.controller.actions(ship_state, game_state)
        # If the user controller returns only 2 values, we add Fire=False, Mine=False
        if len(result) == 2:
            return result[0], result[1], False, False
        return result

def worker_eval(flat_repr, cfg, controller_callback):
    nt, le, ri, m1i, m1ic, m2i, m2ic, rl = flat_repr
    
    # Create Dummy Object with flag to prevent re-compilation
    class DummyChrom:
        def __init__(self): 
            self._flat_repr = flat_repr
            self.is_dummy = True 
            
    dummy = DummyChrom()
    
    scenario = scenarios[cfg["scenario_name"]]
    game = TrainerEnvironment(settings=game_settings)
    total_score = 0.0
    
    for _ in range(cfg["episodes_per_eval"]):
        # 1. Create User Controller (passing dummy)
        raw_controller = controller_callback(dummy)
        # 2. Wrap it to ensure 4 return values
        safe_controller = SafeControllerWrapper(raw_controller)
        
        score, info = game.run(scenario=scenario, controllers=[safe_controller])
        total_score += kessler_score_to_scalar(score, info)
        
    return total_score / cfg["episodes_per_eval"]

def evaluate_population(population, cfg):
    callback = cfg["controller_callback"]
    num_workers = cfg.get("num_workers", 1)
    
    todo_indices = []
    todo_flat_reprs = []
    for i, ind in enumerate(population):
        if not hasattr(ind, "cached_fitness") or ind.cached_fitness is None:
            if not hasattr(ind, "_flat_repr"): compile_chromosome(ind)
            todo_indices.append(i)
            todo_flat_reprs.append(ind._flat_repr)

    results = []
    if num_workers > 1 and len(todo_indices) > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(worker_eval, todo_flat_reprs, [cfg]*len(todo_indices), [callback]*len(todo_indices)))
    else:
        for i in todo_indices:
            ind = population[i]
            results.append(worker_eval(ind._flat_repr, cfg, callback))
            
    for idx, score in zip(todo_indices, results):
        population[idx].cached_fitness = score
    return [ind.cached_fitness for ind in population]

# -----------------------------------------------------------------------------
# SECTION 7: VISUALIZATION UTILS
# -----------------------------------------------------------------------------

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
        color = "#ffcccc"
    elif isinstance(node, FISNode):
        label = f"FIS\n[{node.medium1_center:.2f}, {node.medium2_center:.2f}]"
        color = "#ccffcc"
    else:
        label = "?"
        color = "white"

    G.add_node(nid, label=label, fillcolor=color)
    if parent is not None: G.add_edge(parent, nid)
    if hasattr(node, "left") and node.left: build_graph(node.left, G, nid, counter)
    if hasattr(node, "right") and node.right: build_graph(node.right, G, nid, counter)
    return G, nid

def visualize_tree(root):
    G = nx.DiGraph()
    counter = [0]
    build_graph(root, G, None, counter)

    root_nodes = [n for n in G.nodes if G.in_degree(n) == 0]
    if not root_nodes: return
        
    pos = hierarchy_pos(G, root_nodes[0])
    labels = nx.get_node_attributes(G, "label")
    colors = [nx.get_node_attributes(G, "fillcolor").get(n, "white") for n in G.nodes]

    plt.figure(figsize=(14, 8))
    nx.draw(G, pos, node_color=colors, with_labels=False, arrows=True, node_size=2000, edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold")
    plt.title("Final Fuzzy Tree Structure")
    plt.show()

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

# -----------------------------------------------------------------------------
# SECTION 8: GENETIC ALGORITHM LOOP
# -----------------------------------------------------------------------------

def tournament(pop, fit_dict, k):
    best = random.choice(pop)
    best_f = fit_dict[id(best)]
    for _ in range(k - 1):
        cand = random.choice(pop)
        f = fit_dict[id(cand)]
        if f > best_f: 
            best, best_f = cand, f
    return best

def structural_mutate_constrained(root):
    mutation_type = random.choice(["leaf_swap", "reshuffle"])
    if mutation_type == "leaf_swap":
        leaves = []
        gather_leaf_nodes(root, leaves)
        if len(leaves) >= 2:
            a, b = random.sample(leaves, 2)
            a.index, b.index = b.index, a.index
    elif mutation_type == "reshuffle":
        nodes = []
        def collect(n, p, is_l):
            nodes.append((n, p, is_l))
            if isinstance(n, FISNode):
                if n.left: collect(n.left, n, True)
                if n.right: collect(n.right, n, False)
        collect(root, None, False)
        
        target, parent, is_left = random.choice(nodes)
        indices = []
        gather_leaves(target, indices)
        if len(indices) >= 2:
            new_subtree = random_full_tree_with_leaves(indices)
            if parent is None: return new_subtree
            if is_left: parent.left = new_subtree
            else: parent.right = new_subtree
    return root

def run_ga(cfg):
    print("="*60)
    print(f"Starting GA (Workers: {cfg.get('num_workers', 1)}, Inputs: {cfg['input_count']})")
    print("="*60)
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
    
    popsize, gens = cfg["popsize"], cfg["generations"]
    groups, count, k = [sorted(g) for g in cfg["groups"]], cfg["input_count"], cfg["tournament_k"]
    ga_start_time = time.time()
    max_seconds = cfg.get("max_hours", 2.0) * 3600

    population = [build_initial_tree(count, groups) for _ in range(popsize)]
    for p in population: compile_chromosome(p)
    best_history = []
    
    print(f"{'Gen':<5} | {'Max Fit':<10} | {'Avg Fit':<10} | {'Std Dev':<10} | {'Time':<8} | {'Best Size':<10}")
    print("-" * 75)

    try:
        for gen in range(gens):
            if (time.time() - ga_start_time) >= max_seconds: break
            t0 = time.time()
            fitness_values = evaluate_population(population, cfg)
            eval_time = time.time() - t0

            fit_arr = np.array(fitness_values)
            max_fit, avg_fit, std_fit = np.max(fit_arr), np.mean(fit_arr), np.std(fit_arr)
            best_idx = int(np.argmax(fit_arr))
            current_best_ind = population[best_idx]
            best_history.append(max_fit)

            print(f"{gen:<5} | {max_fit:<10.4f} | {avg_fit:<10.4f} | {std_fit:<10.4f} | {eval_time:<8.2f} | {get_tree_size(current_best_ind):<10}")

            if gen % 10 == 0:
                save_chromosome(current_best_ind, f"checkpoints/gen_{gen}_best.pkl")

            # Elitism
            fit_dict = {id(ind): f for ind, f in zip(population, fitness_values)}
            elite = copy_tree(current_best_ind)
            elite.cached_fitness = max_fit
            compile_chromosome(elite)
            new_pop = [elite]

            # Linear Schedules
            param_cross_prob = linear_schedule(cfg["param_cross_prob_start"], cfg["param_cross_prob_end"], gen, gens)
            struct_mut_prob = linear_schedule(cfg["struct_mut_prob_start"], cfg["struct_mut_prob_end"], gen, gens)
            mf_rate = linear_schedule(cfg["mf_mut_rate_start"], cfg["mf_mut_rate_end"], gen, gens)
            rule_rate = linear_schedule(cfg["rule_mut_rate_start"], cfg["rule_mut_rate_end"], gen, gens)

            while len(new_pop) < popsize:
                p1, p2 = tournament(population, fit_dict, k), tournament(population, fit_dict, k)
                c1, c2 = copy_tree(p1), copy_tree(p2)
                if hasattr(c1, "cached_fitness"): del c1.cached_fitness
                if hasattr(c2, "cached_fitness"): del c2.cached_fitness

                if random.random() < param_cross_prob:
                    A, B = [], []
                    gather_fis_nodes(c1, A)
                    gather_fis_nodes(c2, B)
                    if A and B:
                        na, nb = random.choice(A), random.choice(B)
                        na.medium1_center, nb.medium1_center = nb.medium1_center, na.medium1_center
                        na.medium2_center, nb.medium2_center = nb.medium2_center, na.medium2_center
                        na.rule_constants, nb.rule_constants = nb.rule_constants, na.rule_constants

                if random.random() < struct_mut_prob: c1 = structural_mutate_constrained(c1)
                if random.random() < struct_mut_prob: c2 = structural_mutate_constrained(c2)

                def mutate_params(node):
                    stack = [node]
                    while stack:
                        n = stack.pop()
                        if isinstance(n, FISNode):
                            if random.random() < mf_rate: n.medium1_center += random.uniform(-0.1, 0.1)
                            if random.random() < mf_rate: n.medium2_center += random.uniform(-0.1, 0.1)
                            for i in range(9):
                                if random.random() < rule_rate: n.rule_constants[i] += random.uniform(-0.2, 0.2)
                            if n.left: stack.append(n.left)
                            if n.right: stack.append(n.right)
                
                mutate_params(c1)
                mutate_params(c2)
                clamp_mfs(c1)
                clamp_mfs(c2)
                compile_chromosome(c1)
                compile_chromosome(c2)
                new_pop.append(c1)
                if len(new_pop) < popsize: new_pop.append(c2)
            population = new_pop

    except KeyboardInterrupt:
        print("\n\n*** INTERRUPTED BY USER ***")
        best_idx = int(np.argmax([ind.cached_fitness if hasattr(ind, 'cached_fitness') else -999 for ind in population]))
        save_chromosome(population[best_idx], "interrupted_best.pkl")
        return population[best_idx], best_history

    print("\nGA TRAINING COMPLETE")
    save_chromosome(current_best_ind, "final_best_agent.pkl")
    return current_best_ind, best_history

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Import controller dynamically to avoid circular imports during worker spawn
    try:
        from redone_controller import FuzzyController
    except ImportError:
        print("Error: 'redone_controller.py' not found. Ensure it is in the same directory.")
        exit()

    cfg = get_ga_config()
    cfg["controller_callback"] = FuzzyController

    best, history = run_ga(cfg)
    
    print("\nTraining complete. Saved to 'final_best_agent.pkl'")
    try:
        plot_fitness(history)
        visualize_tree(best)
    except Exception as e:
        print(f"Visualization skipped: {e}")