import numpy as np
import random
import pickle
import copy
import time
import os
import sys
import traceback
import multiprocessing
from datetime import datetime
from numba import njit, float64, int32
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------------------------------------------------------
# SECTION 1: CONFIGURATION
# -----------------------------------------------------------------------------

def get_ga_config():
    cfg = {
        # --- Genetic Algorithm Settings ---
        "popsize": 500,                
        "generations": 500,
        "tournament_k": 4,
        "num_elites": 3,
        
        # --- Multiprocessing ---
        "num_workers": max(1, int(os.cpu_count() * 0.75)),             
        
        # --- Problem Constraints ---
        "input_count": 5,             
        "groups": [],                 
        "max_hours": 12,             

        # --- Mutation Rates ---
        "mf_mut_rate_start": 0.50, "mf_mut_rate_end": 0.02,
        "rule_mut_rate_start": 0.50, "rule_mut_rate_end": 0.02,
        "struct_mut_prob_start": 0.60, "struct_mut_prob_end": 0.10,
        "param_cross_prob_start": 0.80, "param_cross_prob_end": 0.50,
        # Per-node probability during uniform crossover (swap each FIS node pair independently)
        "crossover_per_node_prob": 0.5,

        # --- Parameter Bounds ---
        "rule_const_min": -1.0,
        "rule_const_max":  1.0,

        # --- Stagnation Detection ---
        "stagnation_limit": 20,
        "stagnation_inject_fraction": 0.10,   # Inject 10% fresh randoms on stagnation
        "stagnation_mut_boost": 3.0,           # Temporarily multiply mutation rates by this
        "stagnation_boost_duration": 5,        # How many generations the boost lasts

        # --- Elite Re-evaluation ---
        # Re-evaluate elites every N generations to prevent lucky outliers from persisting
        "elite_reeval_interval": 5,

        # --- Seeding ---
        # List of .pkl file paths to seed the initial population with.
        # Each seed is injected as-is plus (seed_copies - 1) mutated variants.
        "seed_pickles": [],
        "seed_copies": 5,  # Total copies per seed (1 exact + N-1 mutated)

        # --- Evaluation ---
        "scenario_names": ["training1", "training2", "frozen_random"],
        # Scenarios whose scores are shown in the log/plot (excludes noisy random ones)
        "display_scenario_names": ["training1", "training2"],
        "game_type": "TrainerEnvironment",
        "controller_module": "redone_controller", 
        "controller_class": "FuzzyController",
    }
    return cfg

def linear_schedule(start, end, gen, total):
    if total <= 1: return end
    return start + (end - start) * (gen / (total - 1))

# -----------------------------------------------------------------------------
# SECTION 2: FUZZY TREE NODE DEFINITIONS
# -----------------------------------------------------------------------------

class InputNode:
    """Leaf node representing a single normalized input variable (value in [0,1])."""
    def __init__(self, idx):
        self.index = idx
        self.left = None
        self.right = None
    def __repr__(self): return f"InputNode({self.index})"

class FISNode:
    """
    Internal node: a 2-input, 1-output zero-order Takagi-Sugeno FIS.
    
    Each input is fuzzified into 3 MFs (Low, Medium, High) forming a 
    Ruspini partition (they always sum to 1.0). The MF shapes are controlled
    by a single parameter each: medium1_center and medium2_center.
    
    The 9 rule_constants define the consequent for each combination of
    {Low,Med,High} x {Low,Med,High}.
    """
    def __init__(self, rule_min=-1.0, rule_max=1.0):
        self.medium1_center = random.uniform(0.2, 0.8)
        self.medium2_center = random.uniform(0.2, 0.8)
        self.rule_constants = [random.uniform(rule_min, rule_max) for _ in range(9)]
        self.left = None
        self.right = None
    def __repr__(self): return f"FISNode(c1={self.medium1_center:.2f})"

# -----------------------------------------------------------------------------
# SECTION 3: TREE UTILITIES
# -----------------------------------------------------------------------------

def get_tree_size(node):
    """Count total nodes (InputNodes + FISNodes) in the tree."""
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
    """Collect all leaf INPUT INDICES in left-to-right order."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, InputNode): lst.append(n.index)
        else:
            if n.right: stack.append(n.right)
            if n.left: stack.append(n.left)

def gather_leaf_nodes(node, lst):
    """Collect all leaf InputNode OBJECTS."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, InputNode): lst.append(n)
        else:
            if n.right: stack.append(n.right)
            if n.left: stack.append(n.left)

def gather_fis_nodes(node, lst):
    """Collect all FISNode OBJECTS in the tree."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, FISNode):
            lst.append(n)
            if n.right: stack.append(n.right)
            if n.left: stack.append(n.left)

def clamp_params(node, rule_min=-1.0, rule_max=1.0):
    """
    [IMPROVED] Clamp BOTH MF centers AND rule constants.
    
    Original code only clamped MF centers. Unbounded rule constants can
    drift to extreme values through accumulated mutations, causing the
    FIS outputs to saturate or behave erratically.
    """
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, FISNode):
            n.medium1_center = min(max(n.medium1_center, 0.01), 0.99)
            n.medium2_center = min(max(n.medium2_center, 0.01), 0.99)
            for i in range(9):
                n.rule_constants[i] = min(max(n.rule_constants[i], rule_min), rule_max)
            if n.left: stack.append(n.left)
            if n.right: stack.append(n.right)

def copy_tree(node):
    """Deep copy a tree (iterative to avoid stack overflow on deep trees)."""
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

def random_full_tree_with_leaves(indices, rule_min=-1.0, rule_max=1.0):
    """
    Build a random binary tree whose leaves are exactly the given input indices.
    Every internal node is a FISNode (2-in, 1-out FIS).
    """
    leaves = [InputNode(i) for i in indices]
    random.shuffle(leaves)
    nodes = leaves[:]
    while len(nodes) > 1:
        a = nodes.pop(random.randrange(len(nodes)))
        b = nodes.pop(random.randrange(len(nodes)))
        p = FISNode(rule_min, rule_max)
        p.left, p.right = a, b
        nodes.append(p)
    root = nodes[0]
    clamp_params(root, rule_min, rule_max)
    return root

def build_initial_tree(count, groups_sorted, rule_min=-1.0, rule_max=1.0):
    """
    Build an initial tree respecting input groups.
    
    Groups force certain inputs to share a common subtree, which can encode
    domain knowledge (e.g., "these two sensors are related").
    """
    all_inputs = list(range(count))
    used = set()
    subs = []
    for g in groups_sorted:
        valid_g = [x for x in g if x < count and x not in used]
        if valid_g:
            subs.append(random_full_tree_with_leaves(valid_g, rule_min, rule_max))
            for x in valid_g: used.add(x)
    free = [i for i in all_inputs if i not in used]
    if free: subs.append(random_full_tree_with_leaves(free, rule_min, rule_max))
    nodes = subs[:]
    if not nodes: return random_full_tree_with_leaves(all_inputs, rule_min, rule_max)
    while len(nodes) > 1:
        a = nodes.pop(random.randrange(len(nodes)))
        b = nodes.pop(random.randrange(len(nodes)))
        p = FISNode(rule_min, rule_max)
        p.left, p.right = a, b
        nodes.append(p)
    root = nodes[0]
    clamp_params(root, rule_min, rule_max)
    return root

# -----------------------------------------------------------------------------
# SECTION 5: COMPILATION
# -----------------------------------------------------------------------------

def flatten_tree(root):
    """
    Convert a tree of Python objects into flat NumPy arrays for Numba evaluation.
    
    Uses post-order traversal so that every child node has a lower index than
    its parent, guaranteeing correct bottom-up evaluation order.
    """
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
    """
    JIT-compiled batch evaluation of the flattened fuzzy tree.
    
    For each sample in inputs_batch, evaluates the tree bottom-up:
      1. InputNodes look up their value from the input vector.
      2. FISNodes fuzzify both child outputs into Low/Med/High MFs,
         compute all 9 rule activations, and produce a weighted output.
    
    The MFs form a Ruspini partition (Low + Med + High = 1.0 everywhere),
    so the output is always a convex combination of rule constants.
    """
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
                
                if np.isnan(val_a): val_a = 0.0
                if np.isnan(val_b): val_b = 0.0
                
                # Ruspini partition MFs for input A (left child)
                low_a = max(0.0, 1.0 - (val_a * m1_inv[i]))
                high_a = max(0.0, 1.0 - ((1.0 - val_a) * m1_inv_c[i]))
                med_a = max(0.0, 1.0 - low_a - high_a)

                # Ruspini partition MFs for input B (right child)
                low_b = max(0.0, 1.0 - (val_b * m2_inv[i]))
                high_b = max(0.0, 1.0 - ((1.0 - val_b) * m2_inv_c[i]))
                med_b = max(0.0, 1.0 - low_b - high_b)

                # Takagi-Sugeno inference: weighted sum of 9 rule consequents
                r_low_a = (low_b * rules[i, 0]) + (med_b * rules[i, 1]) + (high_b * rules[i, 2])
                r_med_a = (low_b * rules[i, 3]) + (med_b * rules[i, 4]) + (high_b * rules[i, 5])
                r_high_a = (low_b * rules[i, 6]) + (med_b * rules[i, 7]) + (high_b * rules[i, 8])
                
                node_outputs[k_offset + i] = (low_a * r_low_a) + (med_a * r_med_a) + (high_a * r_high_a)

    final_output = np.empty(num_samples, dtype=np.float64)
    for k in range(num_samples):
        final_output[k] = node_outputs[k * num_nodes + (num_nodes - 1)]
    return final_output

def compile_chromosome(chrom):
    if hasattr(chrom, "is_dummy"): return
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

class RedirectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__": module = "algorithms"
        return super().find_class(module, name)

def load_chromosome(filename):
    with open(filename, "rb") as f:
        try:
            ch = pickle.load(f)
        except AttributeError:
            f.seek(0)
            ch = RedirectUnpickler(f).load()
    compile_chromosome(ch)
    return ch

# -----------------------------------------------------------------------------
# SECTION 6: ROBUST WORKER EVALUATION
# -----------------------------------------------------------------------------

import importlib

class SafeControllerWrapper:
    def __init__(self, controller):
        self.controller = controller
        
    @property
    def name(self): return self.controller.name
        
    def actions(self, ship_state, game_state):
        try:
            result = self.controller.actions(ship_state, game_state)
            safe_result = [0.0, 0.0, False, False]
            if np.isfinite(result[0]): safe_result[0] = float(result[0])
            if np.isfinite(result[1]): safe_result[1] = float(result[1])
            if len(result) > 2: safe_result[2] = bool(result[2])
            if len(result) > 3: safe_result[3] = bool(result[3])
            return tuple(safe_result)
        except Exception:
            return 0.0, 0.0, False, False

def kessler_score_to_scalar(score, info):
    try:
        t = score.teams[0]
        return (t.asteroids_hit * t.accuracy) - 20 * t.deaths 
    except:
        return -100.0

def safe_worker_process(payload):
    idx, flat_repr, scenario_names, settings, c_module, c_class, gen_idx = payload
    
    class DummyChrom:
        def __init__(self): 
            self._flat_repr = flat_repr
            self.is_dummy = True 
    dummy = DummyChrom()

    try:
        from kesslergame import KesslerGame, TrainerEnvironment
        from scenarios import scenarios, random_repeatable_frozen
        mod = importlib.import_module(c_module)
        ControllerClass = getattr(mod, c_class)
    except Exception as e:
        return idx, {}, f"Import Error: {e}"

    per_scenario = {}
    
    try:
        for s_name in scenario_names:
            if s_name == "frozen_random":
                scenario = random_repeatable_frozen(gen_idx)
            else:
                scenario = scenarios[s_name]
            game = TrainerEnvironment(settings=settings)
            raw_controller = ControllerClass(dummy)
            safe_controller = SafeControllerWrapper(raw_controller)
            score, info = game.run(scenario=scenario, controllers=[safe_controller])
            per_scenario[s_name] = kessler_score_to_scalar(score, info)
            
        return idx, per_scenario, None
        
    except BaseException as e:
        err_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
        return idx, {}, err_msg

def evaluate_population_robust(population, cfg, gen_idx=0):
    """
    Evaluate the population and store BOTH:
      - cached_fitness:  average over ALL scenarios (used for selection/elitism)
      - display_fitness: average over display_scenario_names only (used for logging/plotting)
    """
    todo_payloads = []
    
    game_settings = {
        "frequency": 30,
        "perf_tracker": False,
        "prints_on": False,
        "graphics_type": 0,
        "realtime_multiplier": 0,
        "time_limit": 120.0,
    }

    scenario_names = cfg["scenario_names"]
    display_names = set(cfg.get("display_scenario_names", scenario_names))

    for i, ind in enumerate(population):
        if not hasattr(ind, "cached_fitness") or ind.cached_fitness is None:
            if not hasattr(ind, "_flat_repr"): compile_chromosome(ind)
            payload = (
                i, 
                ind._flat_repr, 
                scenario_names, 
                game_settings,
                cfg["controller_module"],
                cfg["controller_class"],
                gen_idx
            )
            todo_payloads.append(payload)

    if not todo_payloads:
        return (
            [ind.cached_fitness for ind in population],
            [getattr(ind, "display_fitness", ind.cached_fitness) for ind in population],
        )

    num_workers = cfg.get("num_workers", 1)
    results_map = {}  # idx -> per_scenario dict

    print(f"--> Eval: {len(todo_payloads)} agents using {num_workers} workers...", end=" ", flush=True)

    if num_workers <= 1:
        for payload in todo_payloads:
            res_idx, res_scores, res_err = safe_worker_process(payload)
            if res_err:
                print(f"\n[Agent {res_idx} FAILED]: {res_err}")
            results_map[res_idx] = res_scores
            print(".", end="", flush=True)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            iterator = pool.imap_unordered(safe_worker_process, todo_payloads)
            completed = 0
            total = len(todo_payloads)
            while completed < total:
                try:
                    res_idx, res_scores, res_err = iterator.next(timeout=300)
                    if res_err:
                        print(f"\n[Agent {res_idx} ERROR]: {res_err}")
                        results_map[res_idx] = {}
                    else:
                        results_map[res_idx] = res_scores
                    completed += 1
                    if completed % 5 == 0: print(f"{completed}", end="", flush=True)
                    else: print(".", end="", flush=True)
                except multiprocessing.TimeoutError:
                    print(f"\n[FATAL POOL TIMEOUT] A worker hung indefinitely. Aborting generation.")
                    break
                except StopIteration:
                    break
                except Exception as e:
                    print(f"\n[POOL ERROR] {e}")
                    break

    print(" Done.")

    for i, ind in enumerate(population):
        if i in results_map:
            scores = results_map[i]
            if scores:
                # Full fitness: average over ALL scenarios
                ind.cached_fitness = sum(scores.values()) / len(scores)
                # Display fitness: average over only static/display scenarios
                disp_scores = [v for k, v in scores.items() if k in display_names]
                ind.display_fitness = (sum(disp_scores) / len(disp_scores)) if disp_scores else ind.cached_fitness
            else:
                ind.cached_fitness = -99999.0
                ind.display_fitness = -99999.0
        elif not hasattr(ind, "cached_fitness") or ind.cached_fitness is None:
             ind.cached_fitness = -99999.0
             ind.display_fitness = -99999.0
             
    return (
        [ind.cached_fitness for ind in population],
        [getattr(ind, "display_fitness", ind.cached_fitness) for ind in population],
    )

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
    try:
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
    except Exception as e:
        print(f"Visualization error: {e}")

def plot_fitness(best_history, avg_history=None, cfg=None, save_path=None):
    try:
        generations = len(best_history)
        if generations == 0:
            print("No fitness history to plot.")
            return
        x = np.arange(generations)
        y_best = np.array(best_history)

        fig, ax = plt.subplots(figsize=(14, 7))

        # Best fitness line
        ax.plot(x, y_best, label="Best Fitness (Display)", linewidth=2,
                color="#2196F3", marker='o', markersize=3, markevery=max(1, generations // 30))

        # Average fitness line
        if avg_history and len(avg_history) == generations:
            y_avg = np.array(avg_history)
            ax.plot(x, y_avg, label="Avg Fitness (Display)", linewidth=1.5,
                    color="#FF9800", alpha=0.8)
            # Shaded region between avg and best
            ax.fill_between(x, y_avg, y_best, alpha=0.10, color="#2196F3")

        # Trend line on best
        if generations > 2:
            m, b = np.polyfit(x, y_best, 1)
            yfit = m * x + b
            ax.plot(x, yfit, linestyle="--", color="red", alpha=0.5, linewidth=1,
                    label=f"Trend ({m:+.4f}/gen)")

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Fitness Score (Display Scenarios)", fontsize=12)
        ax.set_title("Evolutionary Progress", fontsize=14, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", fontsize=10)

        # --- Hyperparameter text box ---
        if cfg:
            params = (
                f"pop={cfg.get('popsize','?')}  gens={cfg.get('generations','?')}  "
                f"elites={cfg.get('num_elites','?')}  k={cfg.get('tournament_k','?')}\n"
                f"mf_mut: {cfg.get('mf_mut_rate_start','?')}→{cfg.get('mf_mut_rate_end','?')}  "
                f"rule_mut: {cfg.get('rule_mut_rate_start','?')}→{cfg.get('rule_mut_rate_end','?')}\n"
                f"struct_mut: {cfg.get('struct_mut_prob_start','?')}→{cfg.get('struct_mut_prob_end','?')}  "
                f"cross: {cfg.get('param_cross_prob_start','?')}→{cfg.get('param_cross_prob_end','?')}\n"
                f"rule_bounds: [{cfg.get('rule_const_min','?')}, {cfg.get('rule_const_max','?')}]  "
                f"stag_limit={cfg.get('stagnation_limit','?')}  "
                f"stag_boost={cfg.get('stagnation_mut_boost','?')}×{cfg.get('stagnation_boost_duration','?')}gen\n"
                f"scenarios: {cfg.get('scenario_names','?')}\n"
                f"display:   {cfg.get('display_scenario_names','?')}"
            )
            seeds = cfg.get('seed_pickles', [])
            if seeds:
                seed_names = [os.path.basename(s) for s in seeds]
                params += f"\nseeds: {seed_names} ×{cfg.get('seed_copies','?')}"

            ax.text(0.98, 0.02, params, transform=ax.transAxes,
                    fontsize=7.5, fontfamily="monospace",
                    verticalalignment="bottom", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7))

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved fitness plot to: {save_path}")
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")

# -----------------------------------------------------------------------------
# SECTION 8: GENETIC ALGORITHM LOOP
# -----------------------------------------------------------------------------

def tournament(pop, fit_dict, k):
    """
    Rank-based tournament selection.
    
    Instead of comparing raw fitness (which is noisy due to stochastic game
    evaluation), we compare RANKS. This prevents a single lucky outlier from
    dominating selection — an individual with fitness 500 (lucky) vs 100 (normal)
    would be rank 1 vs rank 2, not 5x more likely to win.
    """
    candidates = random.sample(pop, min(k, len(pop)))
    candidates.sort(key=lambda c: fit_dict.get(id(c), -99999.0), reverse=True)
    return candidates[0]

def structural_mutate_constrained(root, rule_min=-1.0, rule_max=1.0):
    """
    Mutate tree STRUCTURE while preserving the constraint that each input
    index appears exactly once.
    
    Two strategies:
      - leaf_swap: Pick two random leaves and swap their input indices.
      - reshuffle: Pick a random subtree, collect its leaf indices, and
                   rebuild that subtree with a fresh random topology.
    
    [IMPROVED] The internal 'collect' function is now iterative (was recursive).
    [IMPROVED] 'reshuffle' now skips if it randomly selects a single leaf
               (which would be a no-op), improving mutation effectiveness.
    """
    mutation_type = random.choice(["leaf_swap", "reshuffle"])
    
    if mutation_type == "leaf_swap":
        leaves = []
        gather_leaf_nodes(root, leaves)
        if len(leaves) >= 2:
            a, b = random.sample(leaves, 2)
            a.index, b.index = b.index, a.index
            
    elif mutation_type == "reshuffle":
        # [IMPROVED] Iterative collection of (node, parent, is_left_child) tuples
        nodes_info = []
        stack = [(root, None, False)]
        while stack:
            n, p, is_l = stack.pop()
            nodes_info.append((n, p, is_l))
            if isinstance(n, FISNode):
                if n.left: stack.append((n.left, n, True))
                if n.right: stack.append((n.right, n, False))
        
        if not nodes_info: return root
        
        # [IMPROVED] Filter to only FISNodes with 2+ leaves to avoid no-op mutations
        fis_candidates = [(n, p, is_l) for (n, p, is_l) in nodes_info 
                          if isinstance(n, FISNode)]
        
        if fis_candidates:
            target, parent, is_left = random.choice(fis_candidates)
        else:
            return root  # Tree is a single InputNode, nothing to reshuffle
            
        indices = []
        gather_leaves(target, indices)
        if len(indices) >= 2:
            new_subtree = random_full_tree_with_leaves(indices, rule_min, rule_max)
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
    groups = [sorted(g) for g in cfg["groups"]]
    count, k = cfg["input_count"], cfg["tournament_k"]
    num_elites = cfg.get("num_elites", 1)
    rule_min = cfg.get("rule_const_min", -1.0)
    rule_max = cfg.get("rule_const_max", 1.0)
    stagnation_limit = cfg.get("stagnation_limit", 30)
    stagnation_inject = cfg.get("stagnation_inject_fraction", 0.10)
    stagnation_mut_boost = cfg.get("stagnation_mut_boost", 3.0)
    stagnation_boost_duration = cfg.get("stagnation_boost_duration", 5)
    crossover_per_node = cfg.get("crossover_per_node_prob", 0.5)
    elite_reeval_interval = cfg.get("elite_reeval_interval", 5)
    
    ga_start_time = time.time()
    max_seconds = cfg.get("max_hours", 2.0) * 3600

    # --- Build initial population, seeding from pickles if provided ---
    seed_pickles = cfg.get("seed_pickles", [])
    seed_copies = cfg.get("seed_copies", 5)
    population = []

    for pkl_path in seed_pickles:
        if not os.path.isfile(pkl_path):
            print(f"[WARN] Seed pickle not found, skipping: {pkl_path}")
            continue
        try:
            seed_chrom = load_chromosome(pkl_path)
            # 1 exact copy
            exact = copy_tree(seed_chrom)
            compile_chromosome(exact)
            population.append(exact)
            # N-1 mutated variants to spread around the seed's neighborhood
            for _ in range(seed_copies - 1):
                variant = copy_tree(seed_chrom)
                # Light structural + parameter mutation
                if random.random() < 0.3:
                    variant = structural_mutate_constrained(variant, rule_min, rule_max)
                stack = [variant]
                while stack:
                    n = stack.pop()
                    if isinstance(n, FISNode):
                        n.medium1_center += random.gauss(0, 0.08)
                        n.medium2_center += random.gauss(0, 0.08)
                        for i in range(9):
                            n.rule_constants[i] += random.gauss(0, 0.15)
                        if n.left: stack.append(n.left)
                        if n.right: stack.append(n.right)
                clamp_params(variant, rule_min, rule_max)
                compile_chromosome(variant)
                population.append(variant)
            print(f"Seeded {seed_copies} individuals from: {pkl_path}")
        except Exception as e:
            print(f"[WARN] Failed to load seed {pkl_path}: {e}")

    # Fill the rest with random individuals
    num_random = popsize - len(population)
    if num_random > 0:
        for _ in range(num_random):
            ind = build_initial_tree(count, groups, rule_min, rule_max)
            compile_chromosome(ind)
            population.append(ind)
    else:
        # More seeds than popsize — truncate
        population = population[:popsize]

    if seed_pickles:
        print(f"Population: {min(len(seed_pickles)*seed_copies, popsize)} seeded + {max(0, num_random)} random = {len(population)} total")
    best_display_history = []
    avg_display_history = []
    best_ever_display = -99999.0
    stagnation_counter = 0
    boost_remaining = 0  # Countdown for temporary mutation boost
    
    print(f"{'Gen':<5} | {'Disp Fit':<10} | {'Full Fit':<10} | {'Avg Disp':<10} | {'Time':<8} | {'Size':<6} | {'Note':<20}")
    print("-" * 90)

    try:
        for gen in range(gens):
            if (time.time() - ga_start_time) >= max_seconds: 
                print("\n*** TIME LIMIT REACHED ***")
                break
            t0 = time.time()
            note = ""
            
            # --- EVALUATION ---
            # Frozen random scenarios change each generation, so all cached
            # fitness values from the previous generation are stale.
            if "frozen_random" in cfg["scenario_names"]:
                for ind in population:
                    if hasattr(ind, "cached_fitness"):
                        ind.cached_fitness = None

            # Periodically re-evaluate elites to prevent lucky outliers from persisting
            if gen % elite_reeval_interval == 0 and gen > 0:
                for ind in population:
                    if hasattr(ind, "cached_fitness"):
                        ind.cached_fitness = None

            fitness_values, display_values = evaluate_population_robust(population, cfg, gen_idx=gen)
            eval_time = time.time() - t0

            # --- STATS (display fitness for logging, full fitness for selection) ---
            fit_arr = np.array(fitness_values)         # full (all scenarios)
            disp_arr = np.array(display_values)        # display (static scenarios only)
            
            valid_disp = disp_arr[disp_arr > -90000]
            valid_full = fit_arr[fit_arr > -90000]
            
            if len(valid_disp) > 0:
                max_disp, avg_disp = np.max(valid_disp), np.mean(valid_disp)
            else:
                max_disp, avg_disp = -99999.0, -99999.0
            
            max_full = np.max(valid_full) if len(valid_full) > 0 else -99999.0
                
            best_idx = int(np.argmax(fit_arr))  # Best by FULL fitness (for elitism)
            current_best_ind = population[best_idx]
            best_display_history.append(max_disp)
            avg_display_history.append(avg_disp)

            # Stagnation detection based on DISPLAY fitness (stable signal)
            if max_disp > best_ever_display + 0.001:
                best_ever_display = max_disp
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if stagnation_counter >= stagnation_limit:
                note = f"STAG_BOOST+INJECT({stagnation_counter})"
                stagnation_counter = 0
                boost_remaining = stagnation_boost_duration
                # Inject a small number of fresh randoms (replace worst)
                num_inject = max(1, int(popsize * stagnation_inject))
                sorted_indices = np.argsort(fit_arr)
                for j in range(num_inject):
                    replace_idx = sorted_indices[j]
                    population[replace_idx] = build_initial_tree(count, groups, rule_min, rule_max)
                    compile_chromosome(population[replace_idx])
                    population[replace_idx].cached_fitness = None
            elif boost_remaining > 0:
                note = f"BOOST({boost_remaining})"

            print(f"{gen:<5} | {max_disp:<10.4f} | {max_full:<10.4f} | {avg_disp:<10.4f} | {eval_time:<8.2f} | {get_tree_size(current_best_ind):<6} | {note:<20}")

            if gen % 10 == 0:
                save_chromosome(current_best_ind, f"checkpoints/gen_{gen}_best.pkl")

            # --- ELITISM & SELECTION (uses FULL fitness) ---
            fit_dict = {id(ind): f for ind, f in zip(population, fitness_values)}
            
            sorted_pop = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)
            new_pop = []
            for ei in range(min(num_elites, len(sorted_pop))):
                elite = copy_tree(sorted_pop[ei][0])
                # Don't cache elite fitness — they'll be re-evaluated next gen
                # (frozen_random changes, and periodic re-eval catches lucky outliers)
                compile_chromosome(elite)
                new_pop.append(elite)

            # Schedules with optional stagnation boost
            mut_multiplier = stagnation_mut_boost if boost_remaining > 0 else 1.0
            if boost_remaining > 0:
                boost_remaining -= 1
                
            param_cross_prob = linear_schedule(cfg["param_cross_prob_start"], cfg["param_cross_prob_end"], gen, gens)
            struct_mut_prob = min(1.0, linear_schedule(cfg["struct_mut_prob_start"], cfg["struct_mut_prob_end"], gen, gens) * mut_multiplier)
            mf_rate = min(1.0, linear_schedule(cfg["mf_mut_rate_start"], cfg["mf_mut_rate_end"], gen, gens) * mut_multiplier)
            rule_rate = min(1.0, linear_schedule(cfg["rule_mut_rate_start"], cfg["rule_mut_rate_end"], gen, gens) * mut_multiplier)

            # --- OFFSPRING GENERATION ---
            while len(new_pop) < popsize:
                p1 = tournament(population, fit_dict, k)
                p2 = tournament(population, fit_dict, k)
                
                c1, c2 = copy_tree(p1), copy_tree(p2)

                # Uniform crossover: swap params across ALL paired FIS nodes
                if random.random() < param_cross_prob:
                    A, B = [], []
                    gather_fis_nodes(c1, A)
                    gather_fis_nodes(c2, B)
                    if A and B:
                        pairs = min(len(A), len(B))
                        random.shuffle(A)
                        random.shuffle(B)
                        for pi in range(pairs):
                            if random.random() < crossover_per_node:
                                na, nb = A[pi], B[pi]
                                na.medium1_center, nb.medium1_center = nb.medium1_center, na.medium1_center
                                na.medium2_center, nb.medium2_center = nb.medium2_center, na.medium2_center
                                na.rule_constants, nb.rule_constants = nb.rule_constants, na.rule_constants

                # Structural Mutation
                if random.random() < struct_mut_prob: c1 = structural_mutate_constrained(c1, rule_min, rule_max)
                if random.random() < struct_mut_prob: c2 = structural_mutate_constrained(c2, rule_min, rule_max)

                # Gaussian parameter mutation (better local search than uniform)
                def mutate_params(node, mf_r, rule_r):
                    stack = [node]
                    while stack:
                        n = stack.pop()
                        if isinstance(n, FISNode):
                            if random.random() < mf_r: n.medium1_center += random.gauss(0, 0.06)
                            if random.random() < mf_r: n.medium2_center += random.gauss(0, 0.06)
                            for i in range(9):
                                if random.random() < rule_r: n.rule_constants[i] += random.gauss(0, 0.12)
                            if n.left: stack.append(n.left)
                            if n.right: stack.append(n.right)
                
                mutate_params(c1, mf_rate, rule_rate)
                mutate_params(c2, mf_rate, rule_rate)
                clamp_params(c1, rule_min, rule_max)
                clamp_params(c2, rule_min, rule_max)
                compile_chromosome(c1)
                compile_chromosome(c2)
                new_pop.append(c1)
                if len(new_pop) < popsize: new_pop.append(c2)
            
            population = new_pop

    except KeyboardInterrupt:
        print("\n\n*** INTERRUPTED BY USER ***")
    except Exception as e:
        print(f"\n\n*** CRITICAL CRASH: {e} ***")
        traceback.print_exc()

    # --- SAVE ON EXIT ---
    valid_pop = [p for p in population if hasattr(p, 'cached_fitness') and p.cached_fitness is not None]
    if valid_pop:
        best_idx = int(np.argmax([p.cached_fitness for p in valid_pop]))
        final_best = valid_pop[best_idx]
    else:
        final_best = population[0]

    current_date = datetime.now()
    filename = f"final_best_agent_{current_date.month}_{current_date.day}.pkl"
    save_chromosome(final_best, filename)
    print(f"Saved best agent to: {filename}")

    plot_path = f"genetic_algorithm_run_{current_date.month}_{current_date.day}.png"
    plot_fitness(best_display_history, avg_display_history, cfg, save_path=plot_path)
    
    return final_best, best_display_history, avg_display_history

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support() 
    
    import argparse
    parser = argparse.ArgumentParser(description="Fuzzy Tree GA")
    parser.add_argument("--seed", nargs="+", default=[], 
                        help="One or more .pkl files to seed the initial population")
    parser.add_argument("--seed-copies", type=int, default=5,
                        help="Copies per seed (1 exact + N-1 mutated variants)")
    args = parser.parse_args()

    cfg = get_ga_config()
    if args.seed:
        cfg["seed_pickles"] = args.seed
        cfg["seed_copies"] = args.seed_copies
    
    best, best_history, avg_history = run_ga(cfg)
    
    try:
        visualize_tree(best)
    except Exception as e:
        print(f"Viz failed: {e}")


# =============================================================================
# =============================================================================
# =============================================================================
#
#                 DETAILED CODE REPORT & DOCUMENTATION
#
# =============================================================================
# =============================================================================
# =============================================================================
#
# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  FUZZY TREE GENETIC ALGORITHM â€” FULL TECHNICAL REPORT                   â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. OVERVIEW: WHAT THIS CODE DOES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# This code implements a Genetic Algorithm (GA) that simultaneously optimizes
# TWO aspects of a hierarchical fuzzy inference system:
#
#   (A) THE TREE STRUCTURE â€” how individual 2-input, 1-output Fuzzy Inference
#       Systems (FISs) are wired together in a binary tree topology.
#
#   (B) THE FIS PARAMETERS â€” the membership function (MF) shapes and the
#       rule consequent values inside each FIS node.
#
# The result is a single binary tree that takes N normalized inputs (in [0,1])
# and produces a single scalar output. Each internal node is a complete FIS
# that consumes the outputs of its two children, and each leaf node provides
# one of the N raw input values.
#
# KEY CONSTRAINT: Every input index appears exactly once as a leaf. This means
# every input is used, no input is duplicated, and every FIS node receives
# exactly two distinct sub-signals.
#
#
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. DATA STRUCTURES: THE TREE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# The tree is a binary tree built from two node types:
#
#   InputNode(index):
#       - A LEAF node. Has no children.
#       - Represents a single input variable identified by `index`.
#       - When evaluated, it simply returns the value of input[index].
#
#   FISNode:
#       - An INTERNAL node. Always has exactly two children (left, right).
#       - Contains parameters for a complete 2-input, 1-output FIS:
#           * medium1_center (float in [0.01, 0.99]): Controls the MF shape
#             for the LEFT child's output (input A to this FIS).
#           * medium2_center (float in [0.01, 0.99]): Controls the MF shape
#             for the RIGHT child's output (input B to this FIS).
#           * rule_constants (list of 9 floats): The consequent values for
#             each of the 3x3 = 9 fuzzy rules.
#
# Example tree with 5 inputs:
#
#                    FISNode (root)
#                   /              \
#              FISNode              FISNode
#             /      \             /      \
#        FISNode   Input(2)   Input(3)  Input(4)
#        /     \
#   Input(0)  Input(1)
#
# In this tree:
#   - Inputs 0 and 1 are combined by the bottom-left FIS.
#   - That result and Input 2 are combined by the mid-left FIS.
#   - Inputs 3 and 4 are combined by the mid-right FIS.
#   - The two mid-level results are combined by the root FIS.
#   - The root's output is the final tree output.
#
# The tree has N leaves (one per input) and N-1 FIS nodes (a property of
# full binary trees). So for 5 inputs: 5 leaves + 4 FIS nodes = 9 total nodes.
#
#
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. THE FUZZY INFERENCE SYSTEM (FIS) â€” HOW EACH NODE WORKS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Each FISNode implements a zero-order Takagi-Sugeno fuzzy inference system
# with 3 membership functions per input and 9 rules.
#
# 3a. MEMBERSHIP FUNCTIONS (Ruspini Partition)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For each input (val_a from left child, val_b from right child), three
# triangular MFs are computed that ALWAYS sum to exactly 1.0 â€” this is
# called a Ruspini partition:
#
#   Low(x, c)  = max(0, 1 - x/c)
#   High(x, c) = max(0, 1 - (1-x)/(1-c))
#   Med(x, c)  = max(0, 1 - Low - High)   [= whatever is left]
#
# Where c = medium_center (the single tunable parameter per input).
#
# Visually (for c = 0.4):
#
#   1.0 |â•²        â•±â•²        â•±
#       | â•²  Med /  â•² High/
#   0.5 |  â•²   /    â•²   /
#       |Lowâ•² /      â•² /
#   0.0 |â”€â”€â”€â”€â•³â”€â”€â”€â”€â”€â”€â”€â”€â•³â”€â”€â”€â”€â”€â”€
#       0   0.4       1.0
#
# The parameter `c` slides the peak of the Medium MF and controls where
# Low transitions to Medium and Medium transitions to High.
#
# Properties:
#   - Low(0) = 1, Low(c) = 0     â€” peaks at 0, zero at c
#   - High(1) = 1, High(c) = 0   â€” peaks at 1, zero at c
#   - Med(c) = 1                  â€” peaks at c
#   - Low + Med + High = 1.0      â€” always, everywhere (Ruspini partition)
#
# 3b. RULE BASE
# ~~~~~~~~~~~~~~
# The 9 rules cover every combination of {Low, Med, High} for both inputs:
#
#   Rule Index | Input A MF | Input B MF | Consequent
#   -----------|------------|------------|------------------
#       0      |    Low     |    Low     | rule_constants[0]
#       1      |    Low     |    Med     | rule_constants[1]
#       2      |    Low     |    High    | rule_constants[2]
#       3      |    Med     |    Low     | rule_constants[3]
#       4      |    Med     |    Med     | rule_constants[4]
#       5      |    Med     |    High    | rule_constants[5]
#       6      |    High    |    Low     | rule_constants[6]
#       7      |    High    |    Med     | rule_constants[7]
#       8      |    High    |    High    | rule_constants[8]
#
# 3c. INFERENCE (Output Calculation)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The output is a weighted sum (zero-order Takagi-Sugeno style):
#
#   output = Î£_i Î£_j  Î¼_i(a) * Î¼_j(b) * rule_constants[i*3 + j]
#
# Because Î¼_i(a) and Î¼_j(b) each form Ruspini partitions, the weights
# (Î¼_i(a) * Î¼_j(b)) always sum to exactly 1.0. This means the output is
# a CONVEX COMBINATION of the rule constants â€” it is guaranteed to stay
# within [min(rule_constants), max(rule_constants)].
#
# The code computes this efficiently by first aggregating across input B
# for each level of input A, then combining:
#
#   r_low_a  = low_b*r[0] + med_b*r[1] + high_b*r[2]
#   r_med_a  = low_b*r[3] + med_b*r[4] + high_b*r[5]
#   r_high_a = low_b*r[6] + med_b*r[7] + high_b*r[8]
#   output   = low_a*r_low_a + med_a*r_med_a + high_a*r_high_a
#
#
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. COMPILATION: FROM PYTHON OBJECTS TO FAST ARRAYS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# The tree of Python objects (InputNode/FISNode) is compiled into flat NumPy
# arrays for evaluation by a Numba-JIT'd kernel.
#
# flatten_tree() performs a POST-ORDER traversal (left, right, then parent),
# so every child has a lower array index than its parent. This guarantees
# correct bottom-up evaluation: when we process node i, its children at
# indices left[i] and right[i] have already been computed.
#
# The compiled representation consists of 8 arrays:
#   - node_type[i]: 0 = InputNode, 1 = FISNode
#   - left[i], right[i]: For InputNodes, these store the input index.
#                         For FISNodes, these are indices into the flat array.
#   - m1_inv[i], m1_inv_c[i]: Precomputed 1/c1 and 1/(1-c1) to avoid
#                              division in the hot loop.
#   - m2_inv[i], m2_inv_c[i]: Same for the second input's MF.
#   - rules[i, 0..8]: The 9 rule consequent constants.
#
# The Numba kernel (fis_eval_batch) processes multiple input samples in one
# call. It allocates a flat 1D work buffer (num_samples Ã— num_nodes) and
# walks through nodes in order. For InputNodes, it copies the relevant input.
# For FISNodes, it reads its two children's outputs, fuzzifies, applies rules,
# and writes the result. The final output is the last node's value for each
# sample.
#
#
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. THE GENETIC ALGORITHM
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# 5a. CHROMOSOME REPRESENTATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each individual in the GA population IS the tree root node (a FISNode).
# The chromosome encodes both structure (which inputs pair with which) and
# parameters (MF centers + rule constants) simultaneously.
#
# 5b. INITIALIZATION
# ~~~~~~~~~~~~~~~~~~~
# build_initial_tree() creates a random tree:
#   1. Respect any user-defined input groups (inputs that should share a
#      subtree, encoding domain knowledge).
#   2. All remaining free inputs get their own random subtree.
#   3. Sub-trees are then randomly paired until a single root remains.
#
# random_full_tree_with_leaves() takes a list of input indices, shuffles
# them into InputNode leaves, then iteratively picks two random nodes and
# joins them under a new FISNode until one root remains. This produces
# a random binary tree topology.
#
# 5c. SELECTION: TOURNAMENT SELECTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For each offspring, two parents are selected via tournament selection
# with tournament size k (default 3):
#   - Pick k random individuals from the population.
#   - The one with the highest fitness wins.
# This provides selection pressure while maintaining diversity better than
# pure rank selection.
#
# 5d. CROSSOVER: PARAMETER SWAP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# With probability param_cross_prob (starts ~0.8, decays to ~0.5):
#   1. Gather all FISNodes from child1 and child2.
#   2. Pick one random FISNode from each.
#   3. Swap their medium1_center, medium2_center, and rule_constants.
#
# NOTE: This is PARAMETER-ONLY crossover. It does not exchange subtrees
# between parents. Structure mixing relies entirely on structural mutation.
# This is a deliberate design choice to avoid creating trees where inputs
# appear more than once or are missing (which subtree crossover could cause).
#
# 5e. STRUCTURAL MUTATION
# ~~~~~~~~~~~~~~~~~~~~~~~~
# With probability struct_mut_prob (starts ~0.6, decays to ~0.1):
#   Two strategies, chosen randomly:
#
#   "leaf_swap": Pick two random InputNode leaves and swap their indices.
#       Effect: Changes which inputs get paired together, without altering
#       the tree shape. E.g., if FIS_A was processing (Input0, Input1) and
#       FIS_B was processing (Input2, Input3), after swapping Input1â†”Input3,
#       FIS_A now processes (Input0, Input3).
#
#   "reshuffle": Pick a random FIS subtree, collect all its leaf indices,
#       and rebuild that subtree with a completely new random topology.
#       Effect: Reorganizes how a group of inputs are hierarchically combined,
#       potentially changing tree depth and pairing structure. Preserves the
#       exact same set of input indices.
#
# Both strategies preserve the invariant: each input appears exactly once.
#
# 5f. PARAMETER MUTATION
# ~~~~~~~~~~~~~~~~~~~~~~~
# For each FISNode in the tree:
#   - With probability mf_rate: perturb medium1_center by U(-0.1, 0.1)
#   - With probability mf_rate: perturb medium2_center by U(-0.1, 0.1)
#   - For each of the 9 rule constants, with probability rule_rate:
#     perturb by U(-0.2, 0.2)
#
# After mutation, clamp_params() bounds MF centers to [0.01, 0.99] and
# rule constants to [rule_min, rule_max].
#
# 5g. ELITISM
# ~~~~~~~~~~~~
# The top num_elites individuals (default 3) are copied into the next
# generation without modification, preserving their cached fitness so they
# don't need re-evaluation.
#
# 5h. ADAPTIVE RATES (Linear Schedule)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# All mutation/crossover probabilities decay linearly from a high starting
# value to a low ending value over the course of evolution:
#
#   rate(gen) = start + (end - start) * (gen / (total_gens - 1))
#
# This implements an "explore early, exploit late" strategy:
#   - Early generations: High mutation rates â†’ broad search
#   - Late generations: Low mutation rates â†’ fine-tuning best solutions
#
# 5i. STAGNATION DETECTION (NEW)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If the best fitness doesn't improve for `stagnation_limit` consecutive
# generations (default 30), the algorithm injects fresh random individuals
# by replacing the worst 20% of the population. This helps escape local
# optima and is logged in the output as "STAG_RESET".
#
#
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. EVALUATION: HOW FITNESS IS COMPUTED
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Each individual is evaluated by:
#   1. Compiling the tree into flat arrays (if not already compiled).
#   2. Passing those arrays (as a lightweight payload) to a worker process.
#   3. The worker creates a "DummyChrom" object holding the flat arrays.
#   4. The worker dynamically imports the user's FuzzyController class
#      (which knows how to call fuzzy_tree_output internally).
#   5. The controller is wrapped in SafeControllerWrapper to catch any
#      NaN/crash errors and return safe defaults.
#   6. The Kessler game environment runs the scenario with the controller.
#   7. The fitness scalar is computed as:
#        fitness = (asteroids_hit Ã— accuracy) âˆ’ 20 Ã— deaths
#      averaged across all training scenarios.
#
# Multiprocessing uses the "spawn" context for cross-platform safety.
# Workers communicate only primitive data (no complex Python objects),
# avoiding pickle errors from game engine internals.
#
#
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7. ISSUES IDENTIFIED AND IMPROVEMENTS MADE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# ISSUE 1: UNBOUNDED RULE CONSTANTS (BUG â€” Medium Severity)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PROBLEM: The original `clamp_mfs()` only clamped MF centers to [0.01, 0.99]
# but did NOT clamp rule_constants. Through accumulated additive mutations
# (Â±0.2 per generation per rule), constants could drift far beyond [-1, 1]
# over hundreds of generations. While the output is always a convex
# combination of the rule constants (so it won't literally diverge to
# infinity), extreme constant values (e.g., [-50, +50]) make the FIS output
# surface highly irregular and harder to optimize.
#
# FIX: Renamed to `clamp_params()` and added explicit clamping of all 9
# rule_constants to [rule_const_min, rule_const_max] (configurable, default
# [-1, 1]). The bounds are passed through tree construction and mutation.
#
#
# ISSUE 2: SINGLE ELITE (Design â€” Low-Medium Severity)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PROBLEM: Only the single best individual was preserved each generation.
# With stochastic evaluation (game environments have randomness), the best
# individual could be a "lucky" outlier. Single elitism also provides
# minimal population anchor against diversity loss.
#
# FIX: Added configurable multi-elite (default 3). The top N individuals
# are deep-copied into the next generation with their cached fitness.
#
#
# ISSUE 3: NO STAGNATION HANDLING (Design â€” Medium Severity)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PROBLEM: With linearly decaying mutation rates and tournament selection,
# the population could converge prematurely to a local optimum with no
# mechanism to escape. Late-generation low mutation rates would make this
# especially sticky.
#
# FIX: Added stagnation detection. If best fitness fails to improve for
# `stagnation_limit` (default 30) consecutive generations, the worst 20%
# of the population is replaced with fresh random individuals. This injects
# new genetic material and is logged for analysis.
#
#
# ISSUE 4: RECURSIVE collect() IN structural_mutate_constrained (Code Quality)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PROBLEM: Every other tree traversal in the codebase used iterative stacks,
# but the `collect()` helper inside the reshuffle mutation used recursion.
# While unlikely to cause stack overflow with 5 inputs, it's inconsistent
# and could fail with larger input counts.
#
# FIX: Converted to iterative stack-based traversal matching the rest of
# the codebase.
#
#
# ISSUE 5: WASTED MUTATIONS ON LEAF NODES IN RESHUFFLE (Efficiency)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PROBLEM: The original reshuffle could randomly select a single InputNode
# as its target, then `gather_leaves` would find just one index, and the
# `if len(indices) >= 2` guard would skip the mutation entirely. This wasted
# a mutation opportunity â€” the individual was "mutated" but nothing changed.
#
# FIX: The reshuffle now filters candidates to only FISNodes (which always
# have 2+ leaves beneath them), ensuring every reshuffle mutation actually
# changes the tree.
#
#
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 8. POTENTIAL AREAS FOR FUTURE IMPROVEMENT (NOT IMPLEMENTED)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# A. SUBTREE CROSSOVER: The current crossover only swaps FIS parameters
#    between parents but never exchanges structural subtrees. Implementing
#    constraint-preserving subtree crossover (where two subtrees with the
#    same leaf index sets are swapped) could accelerate structural search.
#
# B. FITNESS CACHING / RE-EVALUATION: Because game evaluation is stochastic,
#    re-evaluating the elite periodically (e.g., every 5 generations) with
#    fresh game runs could improve selection reliability by averaging out
#    lucky/unlucky runs.
#
# C. ISLAND MODEL: Running multiple sub-populations independently with
#    periodic migration could explore the search space more broadly.
#
# D. ADAPTIVE MUTATION STEP SIZE: Currently MF perturbation is always
#    U(-0.1, 0.1) and rule perturbation is U(-0.2, 0.2). These could
#    adapt based on fitness improvement rate (self-adaptive mutation).
#
# E. MORE MF PARAMETERS: Each input currently has only one tunable parameter
#    (medium_center). Adding parameters for MF widths or using Gaussian MFs
#    would increase expressiveness at the cost of more parameters to optimize.
#
# =============================================================================
# END OF REPORT
# =============================================================================