import numpy as np
import random
import pickle
import copy
import time
import datetime
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
        print(f"{pad}  c1={node.medium1_center:.3f}, c2={node.medium2_center:.3f}")
        print(f"{pad}  left=")
        print_tree(node.left, indent + 4)
        print(f"{pad}  right=")
        print_tree(node.right, indent + 4)
        print(f"{pad})")


def get_ga_config():
    cfg = {
        "popsize": 50,
        "generations": 100,
        "input_count": 5,           # 5 Inputs
        "groups": [],
        "tournament_k": 3,
        "max_hours": 9.0,

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

        "num_workers": 1,

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
        # Centers of the "Medium" triangle. 
        # Low is always (-inf to c), High is always (c to inf).
        self.medium1_center = random.uniform(0.2, 0.8)
        self.medium2_center = random.uniform(0.2, 0.8)
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
            n.medium1_center = min(max(n.medium1_center, 0.01), 0.99)
            n.medium2_center = min(max(n.medium2_center, 0.01), 0.99)
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
    return root


# -----------------------------------------------------------------------------
# SECTION 5: FLATTEN AND HIGH-PERFORMANCE COMPILE
# -----------------------------------------------------------------------------

def flatten_tree(root):
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
    
    # Pre-calculated inverses for Partition of Unity
    m1_inv = np.zeros(n, dtype=np.float64)        # 1/c1
    m1_inv_c = np.zeros(n, dtype=np.float64)      # 1/(1-c1)
    
    m2_inv = np.zeros(n, dtype=np.float64)        # 1/c2
    m2_inv_c = np.zeros(n, dtype=np.float64)      # 1/(1-c2)
    
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
            
            c1 = node.medium1_center
            c2 = node.medium2_center
            
            # Pre-compute inverses
            m1_inv[i] = 1.0 / c1
            m1_inv_c[i] = 1.0 / (1.0 - c1)
            
            m2_inv[i] = 1.0 / c2
            m2_inv_c[i] = 1.0 / (1.0 - c2)
            
            rules[i, :] = node.rule_constants

    return node_type, left, right, m1_inv, m1_inv_c, m2_inv, m2_inv_c, rules

# -----------------------------------------------------------------------
# OPTIMIZATION: BATCH PROCESSING (PARTITION OF UNITY)
# -----------------------------------------------------------------------

@njit(fastmath=True)
def fis_eval_batch(node_type, left, right, m1_inv, m1_inv_c, m2_inv, m2_inv_c, rules, inputs_batch):
    """
    Evaluates the Fuzzy Tree using Partition of Unity optimization.
    Assumption: Low + Med + High = 1.0.
    Benefit: No division, minimal multiplication.
    """
    num_samples = inputs_batch.shape[0]
    num_nodes = node_type.shape[0]
    
    total_size = num_samples * num_nodes
    node_outputs = np.empty(total_size, dtype=np.float64)
    
    for k in range(num_samples):
        k_offset = k * num_nodes
        
        for i in range(num_nodes):
            if node_type[i] == 0:
                idx = left[i]
                node_outputs[k_offset + i] = inputs_batch[k, idx]
            else:
                val_a = node_outputs[k_offset + left[i]]
                val_b = node_outputs[k_offset + right[i]]
                
                # --- FAST FUZZIFICATION A ---
                # We calculate High and Low. Med is remainder.
                
                # Low A: 1.0 at 0, 0.0 at c1. (Linear drop)
                # Formula: 1 - (val / c1) = 1 - val * inv
                low_a = 1.0 - (val_a * m1_inv[i])
                
                # High A: 0.0 at c1, 1.0 at 1. (Linear rise)
                # Formula: (val - c1) / (1 - c1) = (val * inv_c) - (c1 * inv_c)
                # Simpler: 1 - (1-val)/(1-c1) = 1 - (1-val)*inv_c
                high_a = 1.0 - ((1.0 - val_a) * m1_inv_c[i])
                
                # Branchless Clamp
                if low_a < 0.0: low_a = 0.0
                if high_a < 0.0: high_a = 0.0
                
                # Partition of Unity: Med is whatever is left
                med_a = 1.0 - low_a - high_a
                if med_a < 0.0: med_a = 0.0 # Float error safety

                # --- FAST FUZZIFICATION B ---
                low_b = 1.0 - (val_b * m2_inv[i])
                high_b = 1.0 - ((1.0 - val_b) * m2_inv_c[i])
                
                if low_b < 0.0: low_b = 0.0
                if high_b < 0.0: high_b = 0.0
                
                med_b = 1.0 - low_b - high_b
                if med_b < 0.0: med_b = 0.0

                # --- FACTORED RULE EVALUATION ---
                # Original: Sum( w_ij * R_ij ) / Sum( w_ij )
                # Optimization 1: Sum( w_ij ) is always 1.0 due to Partition of Unity.
                # Optimization 2: Factor out terms.
                # Output = LowA * (Sum of LowA Rules) + MedA * (Sum MedA Rules) ...
                
                # Pre-sum rules weighted by B (Inner Loop)
                # Row 0 (Low A interacting with B)
                r_low_a = (low_b * rules[i, 0]) + (med_b * rules[i, 1]) + (high_b * rules[i, 2])
                
                # Row 1 (Med A interacting with B)
                r_med_a = (low_b * rules[i, 3]) + (med_b * rules[i, 4]) + (high_b * rules[i, 5])
                
                # Row 2 (High A interacting with B)
                r_high_a = (low_b * rules[i, 6]) + (med_b * rules[i, 7]) + (high_b * rules[i, 8])
                
                # Final Sum
                output = (low_a * r_low_a) + (med_a * r_med_a) + (high_a * r_high_a)
                
                node_outputs[k_offset + i] = output

    final_output = np.empty(num_samples, dtype=np.float64)
    for k in range(num_samples):
        final_output[k] = node_outputs[k * num_nodes + (num_nodes - 1)]
        
    return final_output


def compile_chromosome(chrom):
    nt, le, ri, m1i, m1ic, m2i, m2ic, rl = flatten_tree(chrom)
    chrom._flat_repr = (nt, le, ri, m1i, m1ic, m2i, m2ic, rl)
    pass


# -----------------------------------------------------------------------------
# SECTION 6: PUBLIC API
# -----------------------------------------------------------------------------

def fuzzy_tree_output(chrom, *inputs):
    if not hasattr(chrom, "_flat_repr"):
        compile_chromosome(chrom)
        
    nt, le, ri, m1i, m1ic, m2i, m2ic, rl = chrom._flat_repr

    if len(inputs) == 1 and isinstance(inputs[0], np.ndarray):
        arr = inputs[0]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        res = fis_eval_batch(nt, le, ri, m1i, m1ic, m2i, m2ic, rl, arr)
        if res.shape[0] == 1:
            return res[0]
        return res
    else:
        arr = np.array([inputs], dtype=np.float64) 
        res = fis_eval_batch(nt, le, ri, m1i, m1ic, m2i, m2ic, rl, arr)
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
    "graphics_type": 0,
    "realtime_multiplier": 0,
    "time_limit": 120.0,
}

def kessler_score_to_scalar(score, info):
    t = score.teams[0]

    return (t.asteroids_hit * t.accuracy) - 20 * t.deaths 

def fitness(ind, cfg, controller_callback):
    scenario = scenarios[cfg["scenario_name"]]
    episodes = cfg["episodes_per_eval"]
    total = 0.0

    for _ in range(episodes):
        game = TrainerEnvironment(settings=game_settings)
        controller = controller_callback(ind)
        score, info = game.run(scenario=scenario, controllers=[controller])
        total += kessler_score_to_scalar(score, info)

    return -total / episodes

def evaluate_population(population, cfg):
    callback = cfg["controller_callback"]
    results = []
    
    for idx, ind in enumerate(population):
        f = fitness(ind, cfg, callback)
        results.append(f)
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
    print("Starting GA with FAST FUZZY evaluation (Partition of Unity)...")
    popsize = cfg["popsize"]
    gens = cfg["generations"]
    groups = [sorted(g) for g in cfg["groups"]]
    count = cfg["input_count"]
    k = cfg["tournament_k"]
    
    max_seconds = cfg["max_hours"] * 3600 if cfg.get("max_hours") else None
    ga_start_time = time.time()

    population = [build_initial_tree(count, groups) for _ in range(popsize)]
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
            
            c1 = copy_tree(p1)
            c2 = copy_tree(p2)
            
            param_cross_prob = linear_schedule(cfg["param_cross_prob_start"], cfg["param_cross_prob_end"], gen, gens)
            if random.random() < param_cross_prob:
                A = []
                B = []
                gather_fis_nodes(c1, A)
                gather_fis_nodes(c2, B)
                if A and B:
                    na = random.choice(A)
                    nb = random.choice(B)
                    na.medium1_center, nb.medium1_center = nb.medium1_center, na.medium1_center
                    na.medium2_center, nb.medium2_center = nb.medium2_center, na.medium2_center
                    na.rule_constants, nb.rule_constants = nb.rule_constants, na.rule_constants

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

    ga_end_time = time.time()
    total_duration = ga_end_time - ga_start_time
    finish_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "="*60)
    print("GA TRAINING COMPLETE (Fast Fuzzy)")
    print("="*60)
    print(f"Finished at:    {finish_time_str}")
    print(f"Total Runtime:  {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print("-" * 60)
    print("Parameters Used:")
    for key, value in cfg.items():
        print(f"  {key:<25}: {value}")
    print("="*60 + "\n")

    best_index = int(np.argmin(fitness_values))
    return population[best_index], best_history