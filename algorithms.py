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
        "popsize": 10,
        "generations": 150,
        "input_count": 5,           # 5 Inputs. STRICTLY ENFORCED.
        "groups": [],
        "tournament_k": 3,
        "max_hours": 0.03,

        # Parameter Mutation Rates
        "mf_mut_rate_start": 0.50,
        "mf_mut_rate_end":   0.02,

        "rule_mut_rate_start": 0.50,
        "rule_mut_rate_end":   0.02,

        # Structural Mutation Probabilities 
        "struct_mut_prob_start": 0.60, 
        "struct_mut_prob_end":   0.10,

        "param_cross_prob_start": 0.80,
        "param_cross_prob_end":   0.50,

        "struct_cross_prob_start": 0.0, 
        "struct_cross_prob_end":   0.0,

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
    
    # Adding string repr for easier debugging
    def __repr__(self):
        return f"InputNode({self.index})"


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
    """
    Returns a list of integer INDICES. 
    Used by main_train.py for visualization/compatibility.
    """
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

def gather_leaf_nodes(node, lst):
    """
    Returns a list of InputNode OBJECTS.
    Used internally for mutation logic.
    """
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, InputNode):
            lst.append(n)
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
        idx_a = random.randrange(len(nodes))
        a = nodes.pop(idx_a)
        
        idx_b = random.randrange(len(nodes))
        b = nodes.pop(idx_b)
        
        parent = FISNode()
        parent.left = a
        parent.right = b
        nodes.append(parent)

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
    
    if free:
        subs.append(random_full_tree_with_leaves(free))

    nodes = subs[:]
    if not nodes:
        return random_full_tree_with_leaves(all_inputs)

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
    
    m1_inv = np.zeros(n, dtype=np.float64)        
    m1_inv_c = np.zeros(n, dtype=np.float64)      
    
    m2_inv = np.zeros(n, dtype=np.float64)        
    m2_inv_c = np.zeros(n, dtype=np.float64)      
    
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
            
            if c1 < 0.001: c1 = 0.001
            if c1 > 0.999: c1 = 0.999
            if c2 < 0.001: c2 = 0.001
            if c2 > 0.999: c2 = 0.999

            m1_inv[i] = 1.0 / c1
            m1_inv_c[i] = 1.0 / (1.0 - c1)
            
            m2_inv[i] = 1.0 / c2
            m2_inv_c[i] = 1.0 / (1.0 - c2)
            
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
                idx = left[i]
                node_outputs[k_offset + i] = inputs_batch[k, idx]
            else:
                val_a = node_outputs[k_offset + left[i]]
                val_b = node_outputs[k_offset + right[i]]
                
                low_a = 1.0 - (val_a * m1_inv[i])
                high_a = 1.0 - ((1.0 - val_a) * m1_inv_c[i])
                
                if low_a < 0.0: low_a = 0.0
                if high_a < 0.0: high_a = 0.0
                
                med_a = 1.0 - low_a - high_a
                if med_a < 0.0: med_a = 0.0

                low_b = 1.0 - (val_b * m2_inv[i])
                high_b = 1.0 - ((1.0 - val_b) * m2_inv_c[i])
                
                if low_b < 0.0: low_b = 0.0
                if high_b < 0.0: high_b = 0.0
                
                med_b = 1.0 - low_b - high_b
                if med_b < 0.0: med_b = 0.0

                r_low_a = (low_b * rules[i, 0]) + (med_b * rules[i, 1]) + (high_b * rules[i, 2])
                r_med_a = (low_b * rules[i, 3]) + (med_b * rules[i, 4]) + (high_b * rules[i, 5])
                r_high_a = (low_b * rules[i, 6]) + (med_b * rules[i, 7]) + (high_b * rules[i, 8])
                
                output = (low_a * r_low_a) + (med_a * r_med_a) + (high_a * r_high_a)
                node_outputs[k_offset + i] = output

    final_output = np.empty(num_samples, dtype=np.float64)
    for k in range(num_samples):
        final_output[k] = node_outputs[k * num_nodes + (num_nodes - 1)]
        
    return final_output


def compile_chromosome(chrom):
    nt, le, ri, m1i, m1ic, m2i, m2ic, rl = flatten_tree(chrom)
    chrom._flat_repr = (nt, le, ri, m1i, m1ic, m2i, m2ic, rl)


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
    if hasattr(ch, "cached_fitness"):
        del ch.cached_fitness
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
    # ELITISM CACHE
    if hasattr(ind, "cached_fitness") and ind.cached_fitness is not None:
        return ind.cached_fitness

    scenario = scenarios[cfg["scenario_name"]]
    episodes = cfg["episodes_per_eval"]
    total = 0.0

    for _ in range(episodes):
        game = TrainerEnvironment(settings=game_settings)
        controller = controller_callback(ind)
        score, info = game.run(scenario=scenario, controllers=[controller])
        total += kessler_score_to_scalar(score, info)

    val = total / episodes
    ind.cached_fitness = val
    return val

def evaluate_population(population, cfg):
    callback = cfg["controller_callback"]
    results = []
    
    for idx, ind in enumerate(population):
        f = fitness(ind, cfg, callback)
        results.append(f)
    return results


# -----------------------------------------------------------------------------
# SECTION 9: SELECTION AND CONSTRAINED STRUCTURAL MUTATION
# -----------------------------------------------------------------------------

def tournament(pop, fit_dict, k):
    best = random.choice(pop)
    best_f = fit_dict[id(best)]
    for _ in range(k - 1):
        cand = random.choice(pop)
        f = fit_dict[id(cand)]
        if f > best_f: # MAXIMIZATION
            best = cand
            best_f = f
    return best

def collect_nodes_with_parents(node, parent=None, is_left=False, node_list=None):
    if node_list is None:
        node_list = []
    
    node_list.append((node, parent, is_left))
    
    if isinstance(node, FISNode):
        if node.left:
            collect_nodes_with_parents(node.left, node, True, node_list)
        if node.right:
            collect_nodes_with_parents(node.right, node, False, node_list)
    return node_list

def structural_mutate_constrained(root):
    """
    Mutates tree strictly preserving inputs.
    Supported Mutations:
    1. Leaf Swap: Swaps the indices of two leaf nodes.
    2. Subtree Reshuffle: Picks a subtree and rebuilds it randomly using the exact same leaves.
    """
    mutation_type = random.choice(["leaf_swap", "reshuffle"])
    
    if mutation_type == "leaf_swap":
        # USE GATHER_LEAF_NODES (Returns Objects)
        leaves = []
        gather_leaf_nodes(root, leaves)
        
        if len(leaves) >= 2:
            a, b = random.sample(leaves, 2)
            # Swap their input indices
            a.index, b.index = b.index, a.index
            
    elif mutation_type == "reshuffle":
        nodes = collect_nodes_with_parents(root)
        if not nodes: return root
        
        target, parent, is_left = random.choice(nodes)
        
        # USE GATHER_LEAVES (Returns Indices)
        # We need the indices to rebuild a fresh tree
        indices = []
        gather_leaves(target, indices)
        
        if len(indices) < 2:
            return root
            
        new_subtree = random_full_tree_with_leaves(indices)
        
        if parent is None:
            return new_subtree 
        else:
            if is_left:
                parent.left = new_subtree
            else:
                parent.right = new_subtree

    return root

# -----------------------------------------------------------------------------
# SECTION 10: MAIN GA LOOP
# -----------------------------------------------------------------------------

def run_ga(cfg):
    print("Starting GA (Maximization, Strict Input Constraints, Compat Fixed)...")
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
        
        ranked = sorted(zip(fitness_values, population), key=lambda x: x[0], reverse=True)
        
        current_best_fit = ranked[0][0]
        current_best_ind = ranked[0][1]
        
        best_history.append(current_best_fit)
        print(f" Best Fitness: {current_best_fit:.4f}")

        # --- ELITISM ---
        elite = copy_tree(current_best_ind)
        elite.cached_fitness = current_best_fit
        compile_chromosome(elite)
        new_pop = [elite]

        param_cross_prob = linear_schedule(cfg["param_cross_prob_start"], cfg["param_cross_prob_end"], gen, gens)
        struct_mut_prob  = linear_schedule(cfg["struct_mut_prob_start"], cfg["struct_mut_prob_end"], gen, gens)
        mf_rate          = linear_schedule(cfg["mf_mut_rate_start"], cfg["mf_mut_rate_end"], gen, gens)
        rule_rate        = linear_schedule(cfg["rule_mut_rate_start"], cfg["rule_mut_rate_end"], gen, gens)

        while len(new_pop) < popsize:
            p1 = tournament(population, fit_dict, k)
            p2 = tournament(population, fit_dict, k)
            
            c1 = copy_tree(p1)
            c2 = copy_tree(p2)
            
            if hasattr(c1, "cached_fitness"): del c1.cached_fitness
            if hasattr(c2, "cached_fitness"): del c2.cached_fitness

            # Parameter Crossover
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

            # Constrained Structural Mutation
            if random.random() < struct_mut_prob:
                c1 = structural_mutate_constrained(c1)
            if random.random() < struct_mut_prob:
                c2 = structural_mutate_constrained(c2)

            # Parameter Mutation
            def mutate_params(node):
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
            
            mutate_params(c1)
            mutate_params(c2)
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
    print("GA TRAINING COMPLETE (Compat Fixed)")
    print("="*60)
    print(f"Finished at:    {finish_time_str}")
    print(f"Total Runtime:  {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print("-" * 60)
    print("Parameters Used:")
    for key, value in cfg.items():
        print(f"  {key:<25}: {value}")
    print("="*60 + "\n")

    best_index = int(np.argmax(fitness_values))
    return population[best_index], best_history