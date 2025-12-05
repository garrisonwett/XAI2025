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
        "popsize": 2,
        "generations": 2,
        "input_count": 5,           # 5 Inputs
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
    
    # Pre-calculated slopes for branchless math
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
            
            # Pre-compute inverses to avoid division in the hot loop
            # Avoid division by zero by clamping epsilon
            c1 = node.medium1_center
            if c1 < 1e-4: c1 = 1e-4
            if c1 > 0.9999: c1 = 0.9999
            
            c2 = node.medium2_center
            if c2 < 1e-4: c2 = 1e-4
            if c2 > 0.9999: c2 = 0.9999
            
            m1_inv[i] = 1.0 / c1
            m1_inv_c[i] = 1.0 / (1.0 - c1)
            
            m2_inv[i] = 1.0 / c2
            m2_inv_c[i] = 1.0 / (1.0 - c2)
            
            rules[i, :] = node.rule_constants

    # Pre-calculate centers for reference if needed, but we mostly use slopes now
    # We pack everything needed into the tuple
    return node_type, left, right, m1_inv, m1_inv_c, m2_inv, m2_inv_c, rules

# -----------------------------------------------------------------------
# OPTIMIZATION: BATCH PROCESSING (SERIAL OPTIMIZED)
# -----------------------------------------------------------------------

@njit(fastmath=True)
def fis_eval_batch(node_type, left, right, m1_inv, m1_inv_c, m2_inv, m2_inv_c, rules, inputs_batch):
    """
    Evaluates the tree using Branchless Logic and Pre-computed Inverses.
    This is designed to fit entirely in L1/L2 Cache and avoid pipeline stalls.
    """
    num_samples = inputs_batch.shape[0]
    num_nodes = node_type.shape[0]
    
    # 1D flattened array for cache locality
    # Index = k * num_nodes + i
    total_size = num_samples * num_nodes
    node_outputs = np.empty(total_size, dtype=np.float64)
    
    # We assume inputs are type 0 and come first in topological sort usually,
    # but the general logic handles any order (post-order).
    
    for k in range(num_samples):
        # Cache offset for this sample
        k_offset = k * num_nodes
        
        for i in range(num_nodes):
            if node_type[i] == 0:
                # Leaf Node
                idx = left[i]
                node_outputs[k_offset + i] = inputs_batch[k, idx]
            else:
                # FIS Node
                # Fetch children values
                # Post-order guarantees children are already computed at lower indices
                val_a = node_outputs[k_offset + left[i]]
                val_b = node_outputs[k_offset + right[i]]
                
                # --- Branchless Fuzzification A ---
                # Low: Triangle peak 0, ends at c1. (Uses precalc slope 5.0 for -0.2 start)
                # Med: Triangle peak c1.
                # High: Ramp start c1.
                
                # Slope lookup
                inv_c1 = m1_inv[i]
                inv_c1_c = m1_inv_c[i]
                
                # Low Logic: max(0, min((val+0.2)*5, (c1-val)/c1)) -> (c1-val)*inv_c1 = 1 - val*inv_c1
                # Simplified: low is 1 at 0, 0 at c1.
                # Note: Original code had specific -0.2 logic. 
                # (val + 0.2) * 5.0 handles the -0.2 to 0.0 ramp up.
                # (1.0 - val_a * inv_c1) handles the 0.0 to c1 ramp down.
                low1 = (val_a + 0.2) * 5.0
                down_slope = 1.0 - (val_a * inv_c1)
                if down_slope < low1: low1 = down_slope
                if low1 < 0.0: low1 = 0.0
                if low1 > 1.0: low1 = 1.0

                # Med Logic: Triangle 0 -> c1 -> 1
                up = val_a * inv_c1
                down = (1.0 - val_a) * inv_c1_c
                med1 = up
                if down < med1: med1 = down
                if med1 < 0.0: med1 = 0.0
                # med1 doesn't exceed 1.0 mathematically if c1 in (0,1)

                # High Logic: Ramp c1 -> 1
                high1 = (val_a * inv_c1_c) - (inv_c1_c * (1.0 - 1.0/inv_c1_c)) # simpl: (val-c1)/(1-c1)
                # Re-derivation: (val - c1) * inv_1_c1
                # c1 is derived from inv: c1 = 1 / inv_c1? No, simpler to just use pre-calc.
                # Let's use the standard form: (val_a * inv_c1_c) - (c1 * inv_c1_c) 
                # We can approximate or just trust the logic:
                # high = 1 - down_slope_of_med? Yes, exactly.
                # High is just (1 - down_slope_of_med) clipped?
                # Actually, (val - c1)/(1-c1) = 1 - (1-val)/(1-c1) = 1 - down
                high1 = 1.0 - down
                if high1 < 0.0: high1 = 0.0
                if high1 > 1.0: high1 = 1.0

                # --- Branchless Fuzzification B ---
                inv_c2 = m2_inv[i]
                inv_c2_c = m2_inv_c[i]
                
                low2 = (val_b + 0.2) * 5.0
                down_slope2 = 1.0 - (val_b * inv_c2)
                if down_slope2 < low2: low2 = down_slope2
                if low2 < 0.0: low2 = 0.0
                if low2 > 1.0: low2 = 1.0

                up2 = val_b * inv_c2
                down2 = (1.0 - val_b) * inv_c2_c
                med2 = up2
                if down2 < med2: med2 = down2
                if med2 < 0.0: med2 = 0.0

                high2 = 1.0 - down2
                if high2 < 0.0: high2 = 0.0
                if high2 > 1.0: high2 = 1.0

                # --- Rule Evaluation (Manual Unroll + FMA) ---
                # FMA = Fused Multiply Add (conceptually)
                
                num = 0.0
                den = 0.0
                
                # 0: Low/Low
                w = low1 * low2
                num += w * rules[i, 0]
                den += w
                
                # 1: Low/Med
                w = low1 * med2
                num += w * rules[i, 1]
                den += w
                
                # 2: Low/High
                w = low1 * high2
                num += w * rules[i, 2]
                den += w
                
                # 3: Med/Low
                w = med1 * low2
                num += w * rules[i, 3]
                den += w
                
                # 4: Med/Med
                w = med1 * med2
                num += w * rules[i, 4]
                den += w
                
                # 5: Med/High
                w = med1 * high2
                num += w * rules[i, 5]
                den += w
                
                # 6: High/Low
                w = high1 * low2
                num += w * rules[i, 6]
                den += w
                
                # 7: High/Med
                w = high1 * med2
                num += w * rules[i, 7]
                den += w
                
                # 8: High/High
                w = high1 * high2
                num += w * rules[i, 8]
                den += w

                # Final Division & Clamp
                val = 0.0
                if den > 1e-9:
                    val = num / den
                
                if val > 1.0: val = 1.0
                elif val < 0.0: val = 0.0
                
                node_outputs[k_offset + i] = val

    # Return only the root node (last index) for each sample
    # The root is at index (num_nodes - 1)
    # Stride is num_nodes
    # Output array size: num_samples
    final_output = np.empty(num_samples, dtype=np.float64)
    for k in range(num_samples):
        final_output[k] = node_outputs[k * num_nodes + (num_nodes - 1)]
        
    return final_output


def compile_chromosome(chrom):
    # Packs the flat representation with the pre-calculated inverses
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

def kessler_score_to_scalar(score):
    t = score.teams[0]
    return (t.asteroids_hit*t.accuracy) - 20 * t.deaths

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
    print("Starting GA with optimized Numba evaluation...")
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

    best_index = int(np.argmin(fitness_values))
    return population[best_index], best_history