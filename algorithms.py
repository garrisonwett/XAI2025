import numpy as np
import random
import pickle
import copy
from concurrent.futures import ThreadPoolExecutor

from kesslergame import GraphicsType, KesslerGame, TrainerEnvironment
from scenarios import scenarios

############################################################
# GA CONFIG
############################################################

def get_ga_config():
    cfg = {
        "popsize": 30,
        "generations": 50,
        "input_count": 4,
        "groups": [],
        "tournament_k": 3,
        "max_hours": 6.0,  # stop after 3 hours

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

        # controller callback supplied by main_train.py
        "controller_callback": None
    }
    return cfg


def linear_schedule(start, end, gen, total):
    if total <= 1:
        return end
    return start + (end - start) * (gen / (total - 1))

############################################################
# READABLE TREE PRINTING
############################################################

def assign_fis_ids(root):
    counter = [0]
    def visit(node):
        from algorithms import FISNode, InputNode
        if isinstance(node, FISNode):
            node.fis_id = counter[0]
            counter[0] += 1
            visit(node.left)
            visit(node.right)
        elif isinstance(node, InputNode):
            pass
    visit(root)


def print_tree_structure(root, indent=0):
    from algorithms import FISNode, InputNode
    space = "  " * indent
    if isinstance(root, InputNode):
        print(f"{space}Input {root.index}")
        return
    if isinstance(root, FISNode):
        label = f"FIS {getattr(root, 'fis_id', '?')}"
        print(f"{space}{label}")
        print_tree_structure(root.left, indent + 1)
        print_tree_structure(root.right, indent + 1)


def print_membership_functions(root):
    from algorithms import FISNode
    nodes = []
    gather_fis_nodes(root, nodes)

    print("\n===== Membership Functions =====")
    for n in nodes:
        fid = getattr(n, "fis_id", "?")
        print(f"\nFIS {fid}:")
        print(f"  medium1_center: {n.medium1_center:.3f}")
        print(f"  medium2_center: {n.medium2_center:.3f}")
        print("  Triangles:")
        print(f"    Left (low, med, high):")
        print(f"      (-0.2, 0.0, {n.medium1_center:.3f})")
        print(f"      (0.0, {n.medium1_center:.3f}, 1.0)")
        print(f"      ({n.medium1_center:.3f}, 1.0, 1.2)")
        print(f"    Right (low, med, high):")
        print(f"      (-0.2, 0.0, {n.medium2_center:.3f})")
        print(f"      (0.0, {n.medium2_center:.3f}, 1.0)")
        print(f"      ({n.medium2_center:.3f}, 1.0, 1.2)")


def print_rule_constants(root):
    from algorithms import FISNode
    nodes = []
    gather_fis_nodes(root, nodes)

    print("\n===== Rule Constants =====")
    for n in nodes:
        fid = getattr(n, "fis_id", "?")
        print(f"\nFIS {fid}:")
        idx = 0
        for _ in range(3):
            row = []
            for _ in range(3):
                row.append(f"{n.rule_constants[idx]: .3f}")
                idx += 1
            print("  " + "  ".join(row))

############################################################
# GAME SETTINGS (no controller)
############################################################

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

############################################################
# MEMBERSHIP FUNCTION
############################################################

def triangle(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)

############################################################
# TREE NODES
############################################################

class InputNode:
    def __init__(self, idx):
        self.index = idx
        self.left = None
        self.right = None

    def eval(self, vec):
        return vec[self.index]


class FISNode:
    def __init__(self):
        self.medium1_center = random.uniform(0.2, 0.8)
        self.medium2_center = random.uniform(0.2, 0.8)
        self.rule_constants = [random.uniform(-1, 1) for _ in range(9)]
        self.left = None
        self.right = None

    def eval(self, a, b):
        low1  = triangle(a, -0.2, 0.0, self.medium1_center)
        med1  = triangle(a,  0.0, self.medium1_center, 1.0)
        high1 = triangle(a,  self.medium1_center, 1.0, 1.2)

        low2  = triangle(b, -0.2, 0.0, self.medium2_center)
        med2  = triangle(b,  0.0, self.medium2_center, 1.0)
        high2 = triangle(b,  self.medium2_center, 1.0, 1.2)

        L1 = [low1, med1, high1]
        L2 = [low2, med2, high2]

        num = 0.0
        den = 0.0
        idx = 0
        for i in range(3):
            for j in range(3):
                w = L1[i] * L2[j]
                num += w * self.rule_constants[idx]
                den += w
                idx += 1

        if den == 0:
            return 0.0
        return num / den

############################################################
# TREE UTILITIES
############################################################

def copy_tree(node):
    if isinstance(node, InputNode):
        return InputNode(node.index)
    n = FISNode()
    n.medium1_center = node.medium1_center
    n.medium2_center = node.medium2_center
    n.rule_constants = node.rule_constants[:]
    n.left = copy_tree(node.left)
    n.right = copy_tree(node.right)
    return n


def gather_leaves(node, lst):
    if isinstance(node, InputNode):
        lst.append(node.index)
        return
    gather_leaves(node.left, lst)
    gather_leaves(node.right, lst)


def gather_fis_nodes(root, lst):
    if isinstance(root, FISNode):
        lst.append(root)
        gather_fis_nodes(root.left, lst)
        gather_fis_nodes(root.right, lst)


def clamp_mfs(node):
    if isinstance(node, FISNode):
        node.medium1_center = min(max(node.medium1_center, 0.0), 1.0)
        node.medium2_center = min(max(node.medium2_center, 0.0), 1.0)
        clamp_mfs(node.left)
        clamp_mfs(node.right)

############################################################
# RANDOM TREE GENERATION
############################################################

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

############################################################
# TREE EVALUATION
############################################################

def eval_tree(node, x):
    if isinstance(node, InputNode):
        return node.eval(x)
    return node.eval(
        eval_tree(node.left, x),
        eval_tree(node.right, x)
    )


def fuzzy_tree_output(chrom, *inputs):
    arr = np.asarray(inputs, dtype=float)

    leaf_indices = []
    gather_leaves(chrom, leaf_indices)
    if leaf_indices:
        if max(leaf_indices) >= len(arr):
            raise ValueError("Not enough inputs for chromosome")

    return eval_tree(chrom, arr)

############################################################
# SAVE / LOAD
############################################################

def save_chromosome(ch, filename):
    with open(filename, "wb") as f:
        pickle.dump(ch, f)

def load_chromosome(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

############################################################
# FITNESS USING KESSLER
############################################################

def kessler_score_to_scalar(score):
    t = score.teams[0]
    return t.asteroids_hit - 20 * t.deaths


def fitness(ind, cfg, controller_callback):
    scenario = scenarios[cfg["scenario_name"]]
    episodes = cfg["episodes_per_eval"]
    total = 0.0

    for _ in range(episodes):

        if cfg["game_type"] == "KesslerGame":
            game = KesslerGame(settings=game_settings)
        else:
            game = TrainerEnvironment(settings=game_settings)

        controller = controller_callback(ind)
        score, _ = game.run(scenario=scenario, controllers=[controller])

        total += kessler_score_to_scalar(score)

    return -total / episodes


def evaluate_population(population, cfg):
    callback = cfg["controller_callback"]
    if cfg["num_workers"] <= 1:
        return [fitness(ind, cfg, callback) for ind in population]

    with ThreadPoolExecutor(max_workers=cfg["num_workers"]) as ex:
        tasks = ((ind, cfg, callback) for ind in population)
        results = list(ex.map(lambda t: fitness(*t), tasks))
    return results

############################################################
# TOURNAMENT SELECTION
############################################################

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

############################################################
# VALIDATION & INITIALIZATION
############################################################

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

############################################################
# MAIN GA LOOP
############################################################

import time

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

        # Check time limit BEFORE starting a new generation
        if max_seconds is not None:
            elapsed = time.time() - ga_start_time
            if elapsed >= max_seconds:
                print("\nMax time reached. Stopping GA before starting generation", gen)
                break

        gen_start_time = time.time()

        mf_rate = linear_schedule(cfg["mf_mut_rate_start"], cfg["mf_mut_rate_end"], gen, gens)
        rule_rate = linear_schedule(cfg["rule_mut_rate_start"], cfg["rule_mut_rate_end"], gen, gens)

        struct_mut_prob = 0.0
        struct_cross_prob = 0.0
        param_cross_prob = 0.5

        # Evaluation timing
        t0 = time.time()
        fitness_values = evaluate_population(population, cfg)
        eval_time = time.time() - t0
        total_times["eval"] += eval_time

        fit_dict = {id(ind): f for ind, f in zip(population, fitness_values)}
        ranked = sorted(zip(fitness_values, population), key=lambda x: x[0])
        best_fit = ranked[0][0]
        best_history.append(best_fit)

        print("Gen", gen, "best", best_fit)

        # Clone elite
        t_clone_start = time.time()
        elite = copy_tree(ranked[0][1])
        clone_time = time.time() - t_clone_start
        total_times["clone"] += clone_time

        new_pop = [elite]

        while len(new_pop) < popsize:

            # Selection timing
            t_sel = time.time()
            p1 = tournament(population, fit_dict, k)
            p2 = tournament(population, fit_dict, k)
            total_times["selection"] += (time.time() - t_sel)

            t_clone2 = time.time()
            c1 = copy_tree(p1)
            c2 = copy_tree(p2)
            total_times["clone"] += (time.time() - t_clone2)

            # Crossover timing
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

            # Mutation timing
            t_mut = time.time()
            def mutate_params(node):
                if isinstance(node, FISNode):
                    if random.random() < mf_rate:
                        node.medium1_center += random.uniform(-0.1, 0.1)
                    if random.random() < mf_rate:
                        node.medium2_center += random.uniform(-0.1, 0.1)
                    for i in range(9):
                        if random.random() < rule_rate:
                            node.rule_constants[i] += random.uniform(-0.2, 0.2)
                if node.left:
                    mutate_params(node.left)
                if node.right:
                    mutate_params(node.right)

            mutate_params(c1)
            mutate_params(c2)
            total_times["mutation"] += (time.time() - t_mut)

            clamp_mfs(c1)
            clamp_mfs(c2)

            new_pop.append(c1)
            if len(new_pop) < popsize:
                new_pop.append(c2)

        population = new_pop

        gen_total = time.time() - gen_start_time
        total_times["gen_total"] += gen_total

        # Per generation timing
        print(f"Generation {gen} timing:")
        print(f"  eval:      {eval_time:.4f} sec")
        print(f"  clone:     {clone_time:.4f} sec")
        print(f"  gen total: {gen_total:.4f} sec")
        print("")

        # Check time limit AFTER finishing this generation
        if max_seconds is not None:
            elapsed = time.time() - ga_start_time
            if elapsed >= max_seconds:
                print("\nMax time reached. Ending GA after generation", gen)
                break

    # Final summary
    grand_total = total_times["gen_total"]

    print("\n===================================")
    print("Overall GA Timing Breakdown")
    print("===================================")

    for key in ["eval", "selection", "crossover", "mutation", "clone"]:
        sec = total_times[key]
        pct = (sec / grand_total) * 100 if grand_total > 0 else 0
        print(f"{key:10s}: {sec:8.4f} sec   ({pct:5.1f} percent)")

    print(f"\ngrand total: {grand_total:.4f} sec\n")

    # Return best solution found so far
    best_index = int(np.argmin(fitness_values))
    best = population[best_index]
    clamp_mfs(best)
    return best, best_history

