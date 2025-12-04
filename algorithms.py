import numpy as np
import random
import pickle
import copy
import time

# New requirement
from numba import njit

from kesslergame import GraphicsType, KesslerGame, TrainerEnvironment
from scenarios import scenarios


###############################################################################
# SECTION 1: HIGH LEVEL GA CONFIGURATION
###############################################################################

def get_ga_config():
    """
    Return a configuration dictionary for the GA.
    Kept identical to the original public API.
    """
    cfg = {
        "popsize": 10,
        "generations": 10,
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

        "num_workers": 10,

        "scenario_name": "training2",
        "game_type": "TrainerEnvironment",
        "episodes_per_eval": 1,

        "controller_callback": None,
    }
    return cfg


def linear_schedule(start, end, gen, total):
    """
    Linear interpolation between two values.
    Does not alter interface.
    """
    if total <= 1:
        return end
    return start + (end - start) * (gen / (total - 1))


###############################################################################
# SECTION 2: MEMBERSHIP FUNCTION (UNCHANGED)
###############################################################################

def triangle(x, a, b, c):
    """
    Standard triangular membership function.
    """
    if x <= a or x >= c:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


###############################################################################
# SECTION 3: FUZZY TREE NODE DEFINITIONS
###############################################################################

class InputNode:
    """
    Leaf node.
    Stores index of relevant input.
    """
    def __init__(self, idx):
        self.index = idx
        self.left = None
        self.right = None


class FISNode:
    """
    Internal fuzzy inference node.
    Stores two membership centers and nine rule constants.
    """
    def __init__(self):
        self.medium1_center = random.uniform(0.2, 0.8)
        self.medium2_center = random.uniform(0.2, 0.8)
        self.rule_constants = [random.uniform(-1, 1) for _ in range(9)]
        self.left = None
        self.right = None


###############################################################################
# SECTION 4: ITERATIVE TREE UTILITIES (FASTER THAN RECURSION)
###############################################################################

def gather_leaves(node, lst):
    """
    Collect indices of InputNode leaves in an iterative manner.
    Leaves are returned in discovery order.
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


def gather_fis_nodes(node, lst):
    """
    Collect all FISNode objects in the tree.
    Iterative version to avoid slow recursion.
    """
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
    """
    Clamp membership centers to the range [0, 1].
    Iterative for speed.
    """
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
    """
    Deep copy of fuzzy tree without recursion.
    Must preserve exact structure and values.
    """
    if isinstance(node, InputNode):
        return InputNode(node.index)

    # Stack based deep copy
    root_copy = FISNode()
    root_copy.medium1_center = node.medium1_center
    root_copy.medium2_center = node.medium2_center
    root_copy.rule_constants = node.rule_constants[:]

    stack = [(node, root_copy)]

    while stack:
        orig, new = stack.pop()

        # Process left child
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

        # Process right child
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


###############################################################################
# SECTION 5: RANDOM TREE GENERATION
###############################################################################

def random_full_tree_with_leaves(indices):
    """
    Random full binary tree assigning all given input indices to leaves.
    Exactly matches original behavior but faster.
    """
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


###############################################################################
# SECTION 6: TREE VALIDATION
###############################################################################

def validate_tree(root, input_count):
    """
    Ensure tree uses exactly the expected leaf indices.
    """
    leafs = []
    gather_leaves(root, leafs)
    return sorted(leafs) == list(range(input_count))


def build_initial_tree(count, groups_sorted):
    """
    Build initial trees with optional grouping.
    Groups are kept intact.
    """
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


###############################################################################
# SECTION 7: FLATTEN AND COMPILE TREES INTO NUMBA STRUCTURES
###############################################################################

def flatten_tree(root):
    """
    Convert a fuzzy tree into a flat array representation suitable for numba.
    Returns:
        node_type: int array where 0 means input leaf, 1 means FIS node
        left: int array of child indices
        right: int array of child indices
        m1, m2: float arrays of membership centers
        rules: float array (n_nodes, 9)
    """

    # Iterative post-order traversal so children are evaluated before parents
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

    # Assign array index to each node in evaluation order
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


@njit
def fis_eval_numba(node_type, left, right, m1, m2, rules, inputs):
    """
    Numba accelerated evaluation of the fuzzy tree.
    Evaluates all nodes in flat array order.
    node_type: 0 for leaf, 1 for FIS node
    returns scalar output from root.
    """
    n = node_type.shape[0]
    outputs = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if node_type[i] == 0:
            idx = left[i]
            outputs[i] = inputs[idx]
        else:
            a = outputs[left[i]]
            b = outputs[right[i]]

            # Compute membership values for each input
            low1 = 0.0
            med1 = 0.0
            high1 = 0.0
            low2 = 0.0
            med2 = 0.0
            high2 = 0.0

            # Input a
            c1 = m1[i]
            if a > -0.2 and a < c1:
                low1 = (a - (-0.2)) / (0.0 - (-0.2))
            if a > 0.0 and a < c1:
                med1 = (a - 0.0) / (c1 - 0.0)
            if a > c1 and a < 1.0:
                high1 = (a - c1) / (1.0 - c1)

            # Input b
            c2 = m2[i]
            if b > -0.2 and b < c2:
                low2 = (b - (-0.2)) / (0.0 - (-0.2))
            if b > 0.0 and b < c2:
                med2 = (b - 0.0) / (c2 - 0.0)
            if b > c2 and b < 1.0:
                high2 = (b - c2) / (1.0 - c2)

            L1 = np.array([low1, med1, high1])
            L2 = np.array([low2, med2, high2])

            num = 0.0
            den = 0.0
            idx = 0

            for q in range(3):
                for r in range(3):
                    w = L1[q] * L2[r]
                    num += w * rules[i, idx]
                    den += w
                    idx += 1

            if den == 0.0:
                outputs[i] = 0.0
            else:
                outputs[i] = num / den

    return outputs[n - 1]


def compile_chromosome(chrom):
    """
    Build flat representation and compile numba evaluator.
    Attach compiled evaluator to chromosome.
    """
    nt, le, ri, m1, m2, rl = flatten_tree(chrom)

    chrom._flat_repr = (nt, le, ri, m1, m2, rl)
    chrom.compiled = lambda x: fis_eval_numba(nt, le, ri, m1, m2, rl, x)


###############################################################################
# SECTION 8: TREE EVALUATION PUBLIC API (UNCHANGED CALL SIGNATURE)
###############################################################################

def fuzzy_tree_output(chrom, *inputs):
    """
    Public interface for evaluating fuzzy trees.
    Calls the compiled numba function for maximum speed.
    """
    if not hasattr(chrom, "compiled"):
        compile_chromosome(chrom)

    arr = np.asarray(inputs, dtype=np.float64)

    leaf_indices = []
    gather_leaves(chrom, leaf_indices)
    if leaf_indices and max(leaf_indices) >= len(arr):
        raise ValueError("Not enough inputs for chromosome")

    return chrom.compiled(arr)


###############################################################################
# SECTION 9: SAVE AND LOAD
###############################################################################

def save_chromosome(ch, filename):
    with open(filename, "wb") as f:
        pickle.dump(ch, f)


def load_chromosome(filename):
    with open(filename, "rb") as f:
        ch = pickle.load(f)
    compile_chromosome(ch)
    return ch


###############################################################################
# SECTION 10: FITNESS AND GA EVALUATION
###############################################################################

game_settings = {
    "frequency": 30,
    "perf_tracker": False,
    "prints_on": False,
    "graphics_type": GraphicsType.Tkinter,
    "graphics_obj": None,
    "realtime_multiplier": 1.0,
    "time_limit": float("inf"),
    "random_ast_splits": False,
    "UI_settings": {
        "ships": False,
        "lives_remaining": False,
        "accuracy": False,
        "asteroids_hit": False,
        "shots_fired": False,
        "bullets_remaining": False,
        "controller_name": False,
    },
}

def kessler_score_to_scalar(score):
    t = score.teams[0]
    return t.asteroids_hit - 20 * t.deaths

def fitness(ind, cfg, controller_callback):
    scenario = scenarios[cfg["scenario_name"]]
    episodes = cfg["episodes_per_eval"]
    total = 0.0

    for _ in range(episodes):
        game = KesslerGame(settings=game_settings) if cfg["game_type"] == "KesslerGame" else TrainerEnvironment(settings=game_settings)
        controller = controller_callback(ind)
        score, _ = game.run(scenario=scenario, controllers=[controller])
        total += kessler_score_to_scalar(score)

    return -total / episodes


def evaluate_population(population, cfg):
    """
    Sequential population evaluation. 
    Can be expanded later for parallelism if needed.
    """
    callback = cfg["controller_callback"]
    return [fitness(ind, cfg, callback) for ind in population]


###############################################################################
# SECTION 11: SELECTION
###############################################################################

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


###############################################################################
# SECTION 12: MAIN GA LOOP
###############################################################################

def run_ga(cfg):
    popsize = cfg["popsize"]
    gens = cfg["generations"]
    groups = [sorted(g) for g in cfg["groups"]]
    count = cfg["input_count"]
    k = cfg["tournament_k"]
    freeze_gen = cfg["structure_freeze_gen"]

    max_hours = cfg.get("max_hours", None)
    max_seconds = max_hours * 3600 if max_hours is not None else None
    ga_start_time = time.time()

    population = [build_initial_tree(count, groups) for _ in range(popsize)]
    for p in population:
        compile_chromosome(p)

    best_history = []

    total_times = {
        "eval": 0.0,
        "selection": 0.0,
        "crossover": 0.0,
        "mutation": 0.0,
        "clone": 0.0,
        "gen_total": 0.0,
    }

    for gen in range(gens):

        if max_seconds is not None and (time.time() - ga_start_time) >= max_seconds:
            print("Max time reached. Stopping before generation", gen)
            break

        gen_start_time = time.time()

        mf_rate = linear_schedule(cfg["mf_mut_rate_start"], cfg["mf_mut_rate_end"], gen, gens)
        rule_rate = linear_schedule(cfg["rule_mut_rate_start"], cfg["rule_mut_rate_end"], gen, gens)
        param_cross_prob = linear_schedule(cfg["param_cross_prob_start"], cfg["param_cross_prob_end"], gen, gens)

        t0 = time.time()
        fitness_values = evaluate_population(population, cfg)
        eval_time = time.time() - t0
        total_times["eval"] += eval_time

        fit_dict = {id(ind): f for ind, f in zip(population, fitness_values)}
        ranked = sorted(zip(fitness_values, population), key=lambda x: x[0])
        best_fit = ranked[0][0]
        best_history.append(best_fit)

        print("Gen", gen, "best", best_fit)

        t_clone = time.time()
        elite = copy_tree(ranked[0][1])
        compile_chromosome(elite)
        total_times["clone"] += (time.time() - t_clone)

        new_pop = [elite]

        while len(new_pop) < popsize:

            t_sel = time.time()
            p1 = tournament(population, fit_dict, k)
            p2 = tournament(population, fit_dict, k)
            total_times["selection"] += (time.time() - t_sel)

            t_clone2 = time.time()
            c1 = copy_tree(p1)
            c2 = copy_tree(p2)
            total_times["clone"] += (time.time() - t_clone2)

            t_cross = time.time()
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
            total_times["crossover"] += (time.time() - t_cross)

            t_mut = time.time()
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
                        if n.left:
                            stack.append(n.left)
                        if n.right:
                            stack.append(n.right)

            mutate_params(c1)
            mutate_params(c2)
            total_times["mutation"] += (time.time() - t_mut)

            clamp_mfs(c1)
            clamp_mfs(c2)

            compile_chromosome(c1)
            compile_chromosome(c2)

            new_pop.append(c1)
            if len(new_pop) < popsize:
                new_pop.append(c2)

        population = new_pop

        gen_total = time.time() - gen_start_time
        total_times["gen_total"] += gen_total

        print(f"Generation {gen} timing:")
        print(f"  eval:      {eval_time:.4f} sec")
        print(f"  gen total: {gen_total:.4f} sec")
        print("")

        if max_seconds is not None and (time.time() - ga_start_time) >= max_seconds:
            print("Max time reached. Ending GA after generation", gen)
            break

    grand_total = total_times["gen_total"]

    print("\n===================================")
    print("Overall GA Timing Breakdown")
    print("===================================")
    for key in ["eval", "selection", "crossover", "mutation", "clone"]:
        sec = total_times[key]
        pct = (sec / grand_total) * 100 if grand_total > 0 else 0
        print(f"{key:10s}: {sec:8.4f} sec   ({pct:5.1f} percent)")

    print(f"\ngrand total: {grand_total:.4f} sec\n")

    best_index = int(np.argmin(fitness_values))
    best = population[best_index]
    clamp_mfs(best)
    return best, best_history
