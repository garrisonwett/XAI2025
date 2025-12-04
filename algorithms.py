import numpy as np
import random
import pickle
import copy
import time
from numba import njit, float64, int32, void

# -----------------------------------------------------------------------------
# SECTION 1: HIGH LEVEL GA CONFIGURATION
# -----------------------------------------------------------------------------

def print_tree(node, indent=0):
    pad = " " * indent
    if isinstance(node, InputNode):
        print(f"{pad}InputNode(index={node.index})")
        return

    if isinstance(node, FISNode):
        print(f"{pad}FISNode(")
        print(f"{pad}  medium1_center={node.medium1_center:.3f}")
        print(f"{pad}  medium2_center={node.medium2_center:.3f}")
        print(f"{pad}  left=")
        print_tree(node.left, indent + 4)
        print(f"{pad}  right=")
        print_tree(node.right, indent + 4)
        print(f"{pad})")


def get_ga_config():
    """
    Return a configuration dictionary for the GA.
    """
    cfg = {
        "popsize": 50,              # Increased slightly for better diversity
        "generations": 10,          # Increased as speed is now higher
        "input_count": 4,
        "groups": [],
        "tournament_k": 3,
        "max_hours": 6.0,

        "mf_mut_rate_start": 0.90,
        "mf_mut_rate_end":   0.02,

        "rule_mut_rate_start": 0.90,
        "rule_mut_rate_end":   0.02,

        "struct_mut_prob_start": 0.90,
        "struct_mut_prob_end":   0.10,

        "param_cross_prob_start": 0.80,
        "param_cross_prob_end":   0.50,

        "struct_cross_prob_start": 1.00,
        "struct_cross_prob_end":   0.80,

        "structure_freeze_gen": 10,

        "num_workers": 1,         # Keep 1 if using strict determinism or simple debugging

        "scenario_name": "training2",
        "game_type": "TrainerEnvironment",
        "episodes_per_eval": 1,

        "controller_callback": None,
    }
    return cfg


def linear_schedule(start, end, gen, total):
    if total <= 1:
        return end
    return start + (end - start) * (gen / (total - 1))


# -----------------------------------------------------------------------------
# SECTION 2: FUZZY TREE NODE DEFINITIONS
# -----------------------------------------------------------------------------

class InputNode:
    def __init__(self, idx):
        self.index = idx
        self.left = None
        self.right = None


class FISNode:
    def __init__(self):
        self.medium1_center = random.uniform(0.2, 0.8)
        self.medium2_center = random.uniform(0.2, 0.8)
        # 9 rule weights
        self.rule_constants = [random.uniform(-1, 1) for _ in range(9)]
        self.left = None
        self.right = None


# -----------------------------------------------------------------------------
# SECTION 3: ITERATIVE TREE UTILITIES
# -----------------------------------------------------------------------------

def gather_leaves(node, lst):
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, InputNode):
            lst.append(n.index)
        else:
            if n.right is not None:
                stack.append(n.right)
            if n.left is not None:
                stack.append(n.left)


def gather_fis_nodes(node, lst):
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, FISNode):
            lst.append(n)
            if n.right is not None:
                stack.append(n.right)
            if n.left is not None:
                stack.append(n.left)


def clamp_mfs(node):
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, FISNode):
            n.medium1_center = min(max(n.medium1_center, 0.0), 1.0)
            n.medium2_center = min(max(n.medium2_center, 0.0), 1.0)
            if n.left:
                stack.append(n.left)
            if n.right:
                stack.append(n.right)


def copy_tree(node):
    if isinstance(node, InputNode):
        return InputNode(node.index)

    root_copy = FISNode()
    root_copy.medium1_center = node.medium1_center
    root_copy.medium2_center = node.medium2_center
    root_copy.rule_constants = node.rule_constants[:]

    stack = [(node, root_copy)]

    while stack:
        orig, new = stack.pop()

        if orig.left is not None:
            if isinstance(orig.left, InputNode):
                new.left = InputNode(orig.left.index)
            else:
                child = FISNode()
                child.medium1_center = orig.left.medium1_center
                child.medium2_center = orig.left.medium2_center
                child.rule_constants = orig.left.rule_constants[:]
                new.left = child
                stack.append((orig.left, child))

        if orig.right is not None:
            if isinstance(orig.right, InputNode):
                new.right = InputNode(orig.right.index)
            else:
                child = FISNode()
                child.medium1_center = orig.right.medium1_center
                child.medium2_center = orig.right.medium2_center
                child.rule_constants = orig.right.rule_constants[:]
                new.right = child
                stack.append((orig.right, child))

    return root_copy


# -----------------------------------------------------------------------------
# SECTION 4: INITIALIZATION AND VALIDATION
# -----------------------------------------------------------------------------

def random_full_tree_with_leaves(indices):
    leaves = [InputNode(i) for i in indices]
    random.shuffle(leaves)
    nodes = leaves[:]

    while len(nodes) > 1:
        a = nodes.pop(random.randrange(len(nodes)))
        b = nodes.pop(random.randrange(len(nodes)))
        parent = FISNode()
        parent.left = a
        parent.right = b
        nodes.append(parent)

    root = nodes[0]
    clamp_mfs(root)
    return root


def validate_tree(root, input_count):
    leafs = []
    gather_leaves(root, leafs)
    return sorted(leafs) == list(range(input_count))


def build_initial_tree(count, groups_sorted):
    all_inputs = list(range(count))
    grouped = set(sum(groups_sorted, []))
    free = [i for i in all_inputs if i not in grouped]

    subs = []
    for g in groups_sorted:
        subs.append(random_full_tree_with_leaves(g))
    if free:
        subs.append(random_full_tree_with_leaves(free))

    nodes = subs[:]
    while len(nodes) > 1:
        a = nodes.pop(random.randrange(len(nodes)))
        b = nodes.pop(random.randrange(len(nodes)))
        p = FISNode()
        p.left = a
        p.right = b
        nodes.append(p)

    root = nodes[0]
    clamp_mfs(root)
    if not validate_tree(root, count):
        raise RuntimeError("Invalid initial tree")
    return root


# -----------------------------------------------------------------------------
# SECTION 5: FLATTEN AND HIGH-PERFORMANCE COMPILE
# -----------------------------------------------------------------------------

def flatten_tree(root):
    # Standard flattening into arrays
    order = []
    stack = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            order.append(node)
        else:
            stack.append((node, True))
            if isinstance(node, FISNode):
                if node.right is not None:
                    stack.append((node.right, False))
                if node.left is not None:
                    stack.append((node.left, False))

    index_map = {node: i for i, node in enumerate(order)}
    n = len(order)

    node_type = np.zeros(n, dtype=np.int32)
    left = np.zeros(n, dtype=np.int32)
    right = np.zeros(n, dtype=np.int32)
    m1 = np.zeros(n, dtype=np.float64)
    m2 = np.zeros(n, dtype=np.float64)
    rules = np.zeros((n, 9), dtype=np.float64)

    for i, node in enumerate(order):
        if isinstance(node, InputNode):
            node_type[i] = 0
            left[i] = node.index
            right[i] = node.index
        else:
            node_type[i] = 1
            left[i] = index_map[node.left]
            right[i] = index_map[node.right]
            m1[i] = node.medium1_center
            m2[i] = node.medium2_center
            rules[i, :] = node.rule_constants

    return node_type, left, right, m1, m2, rules

# -----------------------------------------------------------------------
# OPTIMIZATION: BATCH PROCESSING (VECTORIZATION)
# -----------------------------------------------------------------------

@njit(fastmath=True)
def fis_eval_batch(node_type, left, right, m1, m2, rules, inputs_batch):
    """
    Evaluates the tree for MULTIPLE inputs at once.
    inputs_batch shape: (Num_Samples, Input_Dim)
    Returns: (Num_Samples,)
    """
    num_samples = inputs_batch.shape[0]
    num_nodes = node_type.shape[0]
    
    # Pre-allocate output matrix for all nodes for all samples
    # This avoids allocation inside the loop
    node_outputs = np.zeros((num_samples, num_nodes), dtype=np.float64)
    
    # Loop over every sample (e.g., every asteroid)
    for k in range(num_samples):
        
        # Process the tree for this single sample
        for i in range(num_nodes):
            if node_type[i] == 0:
                # Leaf Node: Copy input
                idx = left[i]
                node_outputs[k, i] = inputs_batch[k, idx]
            else:
                # FIS Node
                # Fetch inputs from children (already computed due to post-order traversal)
                val_a = node_outputs[k, left[i]]
                val_b = node_outputs[k, right[i]]
                
                # --- Fuzzification (Input A) ---
                c1 = m1[i]
                low1, med1, high1 = 0.0, 0.0, 0.0
                
                if val_a > -0.2 and val_a < c1:
                    low1 = (val_a - (-0.2)) / (c1 - (-0.2)) # Simplified 0.0-(-0.2)
                if val_a > 0.0 and val_a < c1: # Overlap logic
                    pass # Original logic had overlapping triangles, sticking to simple here:
                
                # Re-implementing specific triangle logic from original code accurately:
                if val_a > -0.2 and val_a < 0.0:
                     low1 = (val_a - (-0.2)) / 0.2
                elif val_a >= 0.0 and val_a < c1:
                     low1 = (c1 - val_a) / c1
                     med1 = val_a / c1
                elif val_a >= c1 and val_a < 1.0:
                     med1 = (1.0 - val_a) / (1.0 - c1)
                     high1 = (val_a - c1) / (1.0 - c1)
                elif val_a >= 1.0:
                     high1 = 1.0
                elif val_a <= -0.2:
                     low1 = 1.0

                # --- Fuzzification (Input B) ---
                c2 = m2[i]
                low2, med2, high2 = 0.0, 0.0, 0.0
                
                if val_b > -0.2 and val_b < 0.0:
                     low2 = (val_b - (-0.2)) / 0.2
                elif val_b >= 0.0 and val_b < c2:
                     low2 = (c2 - val_b) / c2
                     med2 = val_b / c2
                elif val_b >= c2 and val_b < 1.0:
                     med2 = (1.0 - val_b) / (1.0 - c2)
                     high2 = (val_b - c2) / (1.0 - c2)
                elif val_b >= 1.0:
                     high2 = 1.0
                elif val_b <= -0.2:
                     low2 = 1.0

                # --- Rule Evaluation ---
                # Manual unrolling prevents creating np.array([low, med, high])
                # which was the major memory killer.
                
                num = 0.0
                den = 0.0
                
                # Rules are flattened 0..8
                # L1 indices: 0=low, 1=med, 2=high
                # L2 indices: 0=low, 1=med, 2=high
                
                # L1 Low
                w = low1 * low2
                num += w * rules[i, 0]
                den += w
                
                w = low1 * med2
                num += w * rules[i, 1]
                den += w
                
                w = low1 * high2
                num += w * rules[i, 2]
                den += w
                
                # L1 Med
                w = med1 * low2
                num += w * rules[i, 3]
                den += w
                
                w = med1 * med2
                num += w * rules[i, 4]
                den += w
                
                w = med1 * high2
                num += w * rules[i, 5]
                den += w
                
                # L1 High
                w = high1 * low2
                num += w * rules[i, 6]
                den += w
                
                w = high1 * med2
                num += w * rules[i, 7]
                den += w
                
                w = high1 * high2
                num += w * rules[i, 8]
                den += w

                if den == 0.0:
                    val = 0.0
                else:
                    val = num / den

                # SAFETY CLAMP: Force value to be within valid range (usually -1 to 1 or 0 to 1)
                # This prevents Infinity/NaN from crashing the game engine
                if val > 1.0: val = 1.0
                elif val < 0.0: val = 0.0
                
                node_outputs[k, i] = val

    # Return the last node (root) output for all samples
    return node_outputs[:, num_nodes - 1]


def compile_chromosome(chrom):
    nt, le, ri, m1, m2, rl = flatten_tree(chrom)
    chrom._flat_repr = (nt, le, ri, m1, m2, rl)
    
    # We allow the compiled function to handle both 1D and 2D arrays
    # by using a wrapper or just relying on the Numba signature
    # Since we want speed, we will assume the User eventually passes 2D.
    # But for backward compatibility with the current Controller, we check.
    pass # No longer attaching lambda to object to avoid pickling issues


# -----------------------------------------------------------------------------
# SECTION 6: PUBLIC API (The Fast Part)
# -----------------------------------------------------------------------------

def fuzzy_tree_output(chrom, *inputs):
    """
    Calculate output.
    Supports two modes:
    1. Standard: fuzzy_tree_output(chrom, arg1, arg2, arg3, arg4)
    2. Batch: fuzzy_tree_output(chrom, numpy_matrix_Nx4)
    """
    if not hasattr(chrom, "_flat_repr"):
        compile_chromosome(chrom)
        
    nt, le, ri, m1, m2, rl = chrom._flat_repr

    # Check if the first input is an array (Batch Mode)
    if len(inputs) == 1 and isinstance(inputs[0], np.ndarray):
        arr = inputs[0]
        # Ensure it is 2D (N, inputs)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        
        res = fis_eval_batch(nt, le, ri, m1, m2, rl, arr)
        
        # If we only asked for 1 item, return float, else return array
        if res.shape[0] == 1:
            return res[0]
        return res
        
    else:
        # Legacy/Scalar Mode (Passed as separate arguments)
        # Convert to a 1-row batch
        arr = np.array([inputs], dtype=np.float64) 
        res = fis_eval_batch(nt, le, ri, m1, m2, rl, arr)
        return res[0]


# -----------------------------------------------------------------------------
# SECTION 7: SAVE AND LOAD
# -----------------------------------------------------------------------------

def save_chromosome(ch, filename):
    if hasattr(ch, "compiled"):
        del ch.compiled
    if hasattr(ch, "_flat_repr"):
        del ch._flat_repr
    with open(filename, "wb") as f:
        pickle.dump(ch, f)

def load_chromosome(filename):
    with open(filename, "rb") as f:
        ch = pickle.load(f)
    compile_chromosome(ch)
    return ch


# -----------------------------------------------------------------------------
# SECTION 8: FITNESS AND GA EVALUATION
# -----------------------------------------------------------------------------

from kesslergame import KesslerGame, TrainerEnvironment
from scenarios import scenarios

game_settings = {
    "frequency": 30,
    "perf_tracker": False,
    "prints_on": False,
    "graphics_type": 0, # NoGraphics
    "realtime_multiplier": 0, # Max speed
    "time_limit": 120,
}

def kessler_score_to_scalar(score):
    t = score.teams[0]
    return (t.asteroids_hit * t.accuracy) - 20 * t.deaths - 100 * t.mean_eval_time

def fitness(ind, cfg, controller_callback):
    scenario = scenarios[cfg["scenario_name"]]
    episodes = cfg["episodes_per_eval"]
    total = 0.0

    for _ in range(episodes):
        game = TrainerEnvironment(settings=game_settings)
        controller = controller_callback(ind)
        score, _ = game.run(scenario=scenario, controllers=[controller])
        total += kessler_score_to_scalar(score)

    return -total / episodes

def evaluate_population(population, cfg):
    callback = cfg["controller_callback"]
    results = []
    
    # We can use a simple loop, or multiprocessing if num_workers > 1
    # For Numba, simple loops are often fine because they release GIL if config correctly,
    # but here we stick to simple serial for stability unless requested.
    
    for idx, ind in enumerate(population):
        t0 = time.time()
        f = fitness(ind, cfg, callback)
        results.append(f)
        
        # Quick check for stalled agents (though improved algo should prevent this)
        if time.time() - t0 > 10.0:
            print(f"Warning: Individual {idx} took >10s")

    return results


# -----------------------------------------------------------------------------
# SECTION 9: SELECTION
# -----------------------------------------------------------------------------

def tournament(pop, fit_dict, k):
    best = random.choice(pop)
    best_f = fit_dict[id(best)]
    for _ in range(k - 1):
        cand = random.choice(pop)
        f = fit_dict[id(cand)]
        if f < best_f:
            best = cand
            best_f = f
    return best


# -----------------------------------------------------------------------------
# SECTION 10: MAIN GA LOOP
# -----------------------------------------------------------------------------

def run_ga(cfg):
    print("Starting GA with optimized Numba evaluation...")
    popsize = cfg["popsize"]
    gens = cfg["generations"]
    groups = [sorted(g) for g in cfg["groups"]]
    count = cfg["input_count"]
    k = cfg["tournament_k"]
    
    max_seconds = cfg["max_hours"] * 3600 if cfg.get("max_hours") else None
    ga_start_time = time.time()

    population = [build_initial_tree(count, groups) for _ in range(popsize)]
    # Pre-compile everyone to warm up Numba cache
    for p in population:
        compile_chromosome(p)

    best_history = []

    for gen in range(gens):
        if max_seconds and (time.time() - ga_start_time) >= max_seconds:
            print("Time limit reached.")
            break

        print(f"--- Gen {gen} ---")
        t0 = time.time()
        
        fitness_values = evaluate_population(population, cfg)
        
        eval_time = time.time() - t0
        print(f" Eval Time: {eval_time:.2f}s")

        fit_dict = {id(ind): f for ind, f in zip(population, fitness_values)}
        ranked = sorted(zip(fitness_values, population), key=lambda x: x[0])
        best_fit = ranked[0][0]
        best_history.append(best_fit)
        
        print(f" Best Fitness: {best_fit:.4f}")

        elite = copy_tree(ranked[0][1])
        compile_chromosome(elite)
        new_pop = [elite]

        while len(new_pop) < popsize:
            p1 = tournament(population, fit_dict, k)
            p2 = tournament(population, fit_dict, k)
            
            # Clone
            c1 = copy_tree(p1)
            c2 = copy_tree(p2)
            
            # Crossover
            param_cross_prob = linear_schedule(cfg["param_cross_prob_start"], cfg["param_cross_prob_end"], gen, gens)
            if random.random() < param_cross_prob:
                A = []
                B = []
                gather_fis_nodes(c1, A)
                gather_fis_nodes(c2, B)
                if A and B:
                    na = random.choice(A)
                    nb = random.choice(B)
                    # Swap internals
                    na.medium1_center, nb.medium1_center = nb.medium1_center, na.medium1_center
                    na.medium2_center, nb.medium2_center = nb.medium2_center, na.medium2_center
                    na.rule_constants, nb.rule_constants = nb.rule_constants, na.rule_constants

            # Mutation
            mf_rate = linear_schedule(cfg["mf_mut_rate_start"], cfg["mf_mut_rate_end"], gen, gens)
            rule_rate = linear_schedule(cfg["rule_mut_rate_start"], cfg["rule_mut_rate_end"], gen, gens)
            
            def mutate(node):
                stack = [node]
                while stack:
                    n = stack.pop()
                    if isinstance(n, FISNode):
                        if random.random() < mf_rate:
                            n.medium1_center += random.uniform(-0.1, 0.1)
                        if random.random() < mf_rate:
                            n.medium2_center += random.uniform(-0.1, 0.1)
                        for i in range(9):
                            if random.random() < rule_rate:
                                n.rule_constants[i] += random.uniform(-0.2, 0.2)
                        if n.left: stack.append(n.left)
                        if n.right: stack.append(n.right)
            
            mutate(c1)
            mutate(c2)
            clamp_mfs(c1)
            clamp_mfs(c2)
            
            compile_chromosome(c1)
            compile_chromosome(c2)
            
            new_pop.append(c1)
            if len(new_pop) < popsize:
                new_pop.append(c2)

        population = new_pop

    best_index = int(np.argmin(fitness_values))
    return population[best_index], best_history