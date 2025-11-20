# =============================================
# TSK FIS Optimization Experiment Framework
# =============================================

import os
import sys
import json
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import logging
from pathlib import Path
from datetime import datetime

# -------------------------------------------------
# Fix Python path so we can import scenarios
# when this file lives in a subfolder like
# "Genetic Algorithm/GA Functions.py"
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from kesslergame import TrainerEnvironment
from scenarios import FROZEN_RANDOM, random_repeatable_frozen, scenarios as SCENARIOS
from TeamTempNameSubmission.fuzzy_controller import FuzzyController
from utils import LoggerUtility, LoggingLevel

# ---------------------------------------------
# CONFIGURATION PARAMETERS (Easy to Adjust)
# ---------------------------------------------
def get_config():
    return {
        "SEED": 42,
        "NUM_TRIALS": 1,
        "NUM_EVAL_EPISODES": 1,
        "POPULATION_SIZE": 30,
        "GENERATIONS": 100,
        "CHROMOSOME_LENGTH": 68,
        "OUTPUT_DIR": "./results/",
        "OPTIMIZERS": ["GA", "PSO", "DE", "RL"],
        "NUM_CORES": 8,

        # Scenario configuration
        # SCENARIO_MODE:
        #   "frozen_random" -> use frozen random layouts from scenarios.py
        #   "static"        -> use a named Scenario from scenarios.py
        "SCENARIO_MODE": "static",
        "TRAINING_SCENARIO_NAME": "training2",
        "REFERENCE_SCENARIO_NAME": "training2",

        # If using frozen_random mode, these control which generations are sampled
        "REFERENCE_SCENARIO_GEN_IDX": 0,

        # GA Parameters
        "GA_MUTATION_RATE": 0.1,
        "GA_CROSSOVER_RATE": 0.7,

        # PSO Parameters
        "PSO_INERTIA": 0.5,
        "PSO_COGNITIVE": 1.5,
        "PSO_SOCIAL": 1.5,

        # DE Parameters
        "DE_MUTATION_FACTOR": 0.8,
        "DE_CROSSOVER_RATE": 0.9,

        # RL Parameters
        "RL_EPISODES": 1000,
    }

CONFIG = get_config()
logger = LoggerUtility(LoggingLevel.DEBUG).get_logger()

# Game settings for all environments
game_settings = {
    "frequency": 30,
    "perf_tracker": True,
    "prints_on": False,
    "graphics_type": None,
    "graphics_obj": None,
    "realtime_multiplier": 1,
    "time_limit": float("inf"),
    "random_ast_splits": False,
    "UI_settings": {},
}


def create_game_env():
    return TrainerEnvironment(settings=game_settings)


# -------------------------------------------------
# Scenario selection helpers
# -------------------------------------------------
def get_training_scenario(gen_idx: int) -> "Scenario":
    """
    Scenario used for training or search at a given generation or episode.
    Controlled by CONFIG["SCENARIO_MODE"].

    - "static": always use a named scenario from scenarios.py
    - "frozen_random": use frozen random layouts keyed by gen_idx
    """
    mode = CONFIG.get("SCENARIO_MODE", "frozen_random")
    if mode == "static":
        name = CONFIG.get("TRAINING_SCENARIO_NAME", "training2")
        if name not in SCENARIOS:
            raise KeyError(f"Training scenario '{name}' not found in scenarios.scenarios")
        return SCENARIOS[name]
    elif mode == "frozen_random":
        return random_repeatable_frozen(gen_idx)
    else:
        raise ValueError(f"Unknown SCENARIO_MODE: {mode}")


def get_reference_scenario() -> "Scenario":
    """
    Scenario used to evaluate the best solution of each generation or episode,
    so that the fitness curves are on a consistent benchmark.

    Respect CONFIG["SCENARIO_MODE"]:
    - "static": use CONFIG["REFERENCE_SCENARIO_NAME"] from scenarios.scenarios
    - "frozen_random": use FROZEN_RANDOM with fixed gen index
    """
    mode = CONFIG.get("SCENARIO_MODE", "frozen_random")
    if mode == "static":
        name = CONFIG.get("REFERENCE_SCENARIO_NAME", "training2")
        if name not in SCENARIOS:
            raise KeyError(f"Reference scenario '{name}' not found in scenarios.scenarios")
        return SCENARIOS[name]
    elif mode == "frozen_random":
        gen_idx = CONFIG.get("REFERENCE_SCENARIO_GEN_IDX", 0)
        return FROZEN_RANDOM.get(gen_idx=gen_idx)
    else:
        raise ValueError(f"Unknown SCENARIO_MODE: {mode}")


def get_final_eval_scenario() -> "Scenario":
    """
    Scenario used for final evaluation of each trial and for cross optimizer comparison.

    - "static": same as reference scenario
    - "frozen_random": use a fixed but separate frozen random generation
    """
    mode = CONFIG.get("SCENARIO_MODE", "frozen_random")
    if mode == "static":
        # For static mode, just reuse the reference scenario for final evaluation
        return get_reference_scenario()
    elif mode == "frozen_random":
        # Keep these large gen indices for separation from training and reference
        return FROZEN_RANDOM.get(gen_idx=9999)
    else:
        raise ValueError(f"Unknown SCENARIO_MODE: {mode}")


def get_final_comparison_scenario() -> "Scenario":
    """
    Scenario used in final_comparison across optimizers.

    - "static": same as reference scenario
    - "frozen_random": another distinct frozen random layout
    """
    mode = CONFIG.get("SCENARIO_MODE", "frozen_random")
    if mode == "static":
        return get_reference_scenario()
    elif mode == "frozen_random":
        return FROZEN_RANDOM.get(gen_idx=99999)
    else:
        raise ValueError(f"Unknown SCENARIO_MODE: {mode}")


# -------------------------------------------------
# Fitness evaluation
# -------------------------------------------------
# Top level worker function so it is picklable by multiprocessing
def _run_episode_for_chromosome(args):
    """
    Helper for a single episode.

    args: tuple (chromosome, scenario)
    """
    chromosome, scenario = args
    game = create_game_env()
    score, _ = game.run(chromosome, scenario=scenario, controllers=[FuzzyController()])
    team = score.teams[0]
    fitness = (team.asteroids_hit * team.accuracy) - team.deaths * 50
    return fitness


def evaluate_chromosome(chromosome, scenario):
    """
    Evaluate a chromosome by running several episodes in the same scenario
    and averaging the fitness. Single process version.
    """
    scores = [
        _run_episode_for_chromosome((chromosome, scenario))
        for _ in range(CONFIG["NUM_EVAL_EPISODES"])
    ]
    return sum(scores) / len(scores)


def fitness_function(args):
    """
    Wrapper used by pool.map.

    args: (chromosome, scenario)
    """
    chromosome, scenario = args
    return evaluate_chromosome(chromosome, scenario)


# -------------------------------------------------
# Genetic Algorithm
# -------------------------------------------------
def run_ga(chromosome_length, pop_size, generations, pool):
    mutation_rate = CONFIG["GA_MUTATION_RATE"]
    crossover_rate = CONFIG["GA_CROSSOVER_RATE"]
    population = np.random.rand(pop_size, chromosome_length)
    fitness_curve = []

    reference_scenario = get_reference_scenario()

    for gen in range(generations):
        # Scenario used for search this generation
        search_scenario = get_training_scenario(gen)

        # Evaluate population on search scenario in parallel
        args_list = [(ind, search_scenario) for ind in population]
        if pool is not None:
            fitness = np.array(pool.map(fitness_function, args_list))
        else:
            fitness = np.array([fitness_function(args) for args in args_list])

        elite = population[np.argmax(fitness)]
        new_population = [elite.copy()]

        # Selection, crossover, mutation to form new population
        for _ in range(pop_size - 1):
            idx = np.argsort(fitness)[-2:]
            p1, p2 = population[idx[0]], population[idx[1]]
            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, chromosome_length - 1)
                child = np.concatenate([p1[:point], p2[point:]])
            else:
                child = p1.copy()
            for i in range(chromosome_length):
                if np.random.rand() < mutation_rate:
                    child[i] = np.random.rand()
            new_population.append(child)

        population = np.clip(np.array(new_population), 0, 1)

        # Track progress by evaluating the elite on the fixed reference scenario
        reference_fitness = evaluate_chromosome(elite, reference_scenario)
        fitness_curve.append(reference_fitness)
        print(f"[{datetime.now()}] GA Gen {gen + 1}: Ref Fitness = {reference_fitness:.4f}")

    # Final selection on a fixed scenario for consistency
    final_scenario = get_final_eval_scenario()
    args_list = [(ind, final_scenario) for ind in population]
    if pool is not None:
        final_fitness = np.array(pool.map(fitness_function, args_list))
    else:
        final_fitness = np.array([fitness_function(args) for args in args_list])

    best_idx = np.argmax(final_fitness)
    return population[best_idx].tolist(), {"fitness_curve": fitness_curve}


# -------------------------------------------------
# Particle Swarm Optimization
# -------------------------------------------------
def run_pso(chromosome_length, pop_size, generations, pool):
    w, c1, c2 = CONFIG["PSO_INERTIA"], CONFIG["PSO_COGNITIVE"], CONFIG["PSO_SOCIAL"]
    positions = np.random.rand(pop_size, chromosome_length)
    velocities = np.random.rand(pop_size, chromosome_length) * 0.1
    personal_best = positions.copy()

    # Initial search scenario for seeding personal bests
    initial_search_scenario = get_training_scenario(0)
    args_list = [(ind, initial_search_scenario) for ind in positions]
    if pool is not None:
        personal_best_scores = np.array(pool.map(fitness_function, args_list))
    else:
        personal_best_scores = np.array([fitness_function(args) for args in args_list])

    global_best = personal_best[np.argmax(personal_best_scores)]
    global_best_score = max(personal_best_scores)

    reference_scenario = get_reference_scenario()
    # Track progress on reference scenario
    initial_reference_fitness = evaluate_chromosome(global_best, reference_scenario)
    fitness_curve = [initial_reference_fitness]
    print(f"[{datetime.now()}] PSO Init: Ref Fitness = {initial_reference_fitness:.4f}")

    for gen in range(generations):
        search_scenario = get_training_scenario(gen + 1)

        # Update positions and velocities
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best[i] - positions[i])
                + c2 * r2 * (global_best - positions[i])
            )
            positions[i] = np.clip(positions[i] + velocities[i], 0, 1)

        # Evaluate all particles on this generation's search scenario in parallel
        args_list = [(positions[i], search_scenario) for i in range(pop_size)]
        if pool is not None:
            scores = np.array(pool.map(fitness_function, args_list))
        else:
            scores = np.array([fitness_function(args) for args in args_list])

        # Update personal and global bests
        for i in range(pop_size):
            score = scores[i]
            if score > personal_best_scores[i]:
                personal_best[i] = positions[i].copy()
                personal_best_scores[i] = score
                if score > global_best_score:
                    global_best = positions[i].copy()
                    global_best_score = score

        # Log fitness on the reference scenario using current global best
        reference_fitness = evaluate_chromosome(global_best, reference_scenario)
        fitness_curve.append(reference_fitness)
        print(f"[{datetime.now()}] PSO Gen {gen + 1}: Ref Fitness = {reference_fitness:.4f}")

    return global_best.tolist(), {"fitness_curve": fitness_curve}


# -------------------------------------------------
# Differential Evolution
# -------------------------------------------------
def run_de(chromosome_length, pop_size, generations, pool):
    F = CONFIG["DE_MUTATION_FACTOR"]
    CR = CONFIG["DE_CROSSOVER_RATE"]
    population = np.random.rand(pop_size, chromosome_length)
    fitness_curve = []

    reference_scenario = get_reference_scenario()

    for gen in range(generations):
        search_scenario = get_training_scenario(gen)

        # Evaluate current population in parallel on search scenario
        args_list = [(ind, search_scenario) for ind in population]
        if pool is not None:
            fitness = np.array(pool.map(fitness_function, args_list))
        else:
            fitness = np.array([fitness_function(args) for args in args_list])

        best_idx = np.argmax(fitness)
        best_individual = population[best_idx].copy()
        new_population = [best_individual]

        # For each target vector, build a trial vector
        for i in range(pop_size - 1):
            # Indices of all individuals except i
            all_indices = np.arange(pop_size)
            mask = all_indices != i
            candidate_indices = all_indices[mask]

            if candidate_indices.size >= 3:
                # Standard DE: 3 distinct vectors different from target
                a_idx, b_idx, c_idx = np.random.choice(candidate_indices, 3, replace=False)
            else:
                # Population is too small to draw 3 distinct others
                # Fall back to sampling with replacement from full pool
                a_idx, b_idx, c_idx = np.random.choice(all_indices, 3, replace=True)

            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            mutant = np.clip(a + F * (b - c), 0, 1)
            cross_points = np.random.rand(chromosome_length) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, chromosome_length)] = True
            trial = np.where(cross_points, mutant, population[i])

            # Evaluate trial fitness on search scenario (single process)
            trial_fitness = evaluate_chromosome(trial, search_scenario)
            if trial_fitness > fitness[i]:
                new_population.append(trial)
            else:
                new_population.append(population[i])

        population = np.array(new_population)

        # After updating population, evaluate again for logging
        args_list = [(ind, search_scenario) for ind in population]
        if pool is not None:
            fitness = np.array(pool.map(fitness_function, args_list))
        else:
            fitness = np.array([fitness_function(args) for args in args_list])

        best_idx = np.argmax(fitness)
        best_individual = population[best_idx]

        reference_fitness = evaluate_chromosome(best_individual, reference_scenario)
        fitness_curve.append(reference_fitness)
        print(f"[{datetime.now()}] DE Gen {gen + 1}: Ref Fitness = {reference_fitness:.4f}")

    best_idx = np.argmax(fitness)
    return population[best_idx].tolist(), {"fitness_curve": fitness_curve}


# -------------------------------------------------
# Simple Random Search (RL placeholder)
# -------------------------------------------------
def run_rl(chromosome_length, episodes):
    """
    Simple random search that keeps track of the best seen chromosome.
    Fitness curve is logged on the fixed reference scenario.
    Single process only.
    """
    best = None
    best_score_search = -np.inf
    fitness_curve = []

    reference_scenario = get_reference_scenario()

    for ep in range(episodes):
        candidate = np.random.rand(chromosome_length)
        # Search scenario still changes with episode if using frozen_random
        search_scenario = get_training_scenario(ep)
        score = evaluate_chromosome(candidate, search_scenario)
        if score > best_score_search or best is None:
            best_score_search = score
            best = candidate.copy()

        # Log best so far on the fixed reference scenario
        reference_fitness = evaluate_chromosome(best, reference_scenario)
        fitness_curve.append(reference_fitness)
        print(f"[{datetime.now()}] RL Episode {ep + 1}: Ref Fitness = {reference_fitness:.4f}")

    return best.tolist(), {"fitness_curve": fitness_curve}


# -------------------------------------------------
# Trial runner and orchestration
# -------------------------------------------------
def run_single_trial(args):
    optimizer_name, trial_index = args
    np.random.seed(CONFIG["SEED"] + trial_index)
    print(f"[{datetime.now()}] Running {optimizer_name} trial {trial_index + 1}")

    with mp.Pool(CONFIG["NUM_CORES"]) as pool:
        if optimizer_name == "GA":
            best, log = run_ga(
                CONFIG["CHROMOSOME_LENGTH"],
                CONFIG["POPULATION_SIZE"],
                CONFIG["GENERATIONS"],
                pool,
            )
        elif optimizer_name == "PSO":
            best, log = run_pso(
                CONFIG["CHROMOSOME_LENGTH"],
                CONFIG["POPULATION_SIZE"],
                CONFIG["GENERATIONS"],
                pool,
            )
        elif optimizer_name == "DE":
            best, log = run_de(
                CONFIG["CHROMOSOME_LENGTH"],
                CONFIG["POPULATION_SIZE"],
                CONFIG["GENERATIONS"],
                pool,
            )
        elif optimizer_name == "RL":
            # RL uses no multiprocessing internally
            best, log = run_rl(
                CONFIG["CHROMOSOME_LENGTH"],
                CONFIG["RL_EPISODES"],
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Final evaluation on a fixed scenario, single process
    final_scenario = get_final_eval_scenario()
    final_score = evaluate_chromosome(best, final_scenario)
    return final_score, best, log


def run_optimizer_trials(optimizer_name):
    results = [run_single_trial((optimizer_name, i)) for i in range(CONFIG["NUM_TRIALS"])]
    return zip(*results)


def save_results(optimizer_name, scores, chromosomes, logs):
    Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(CONFIG["OUTPUT_DIR"], f"{optimizer_name}_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Trial", "Score"])
        for i, s in enumerate(scores):
            writer.writerow([i + 1, s])

    json.dump(
        list(chromosomes),
        open(os.path.join(CONFIG["OUTPUT_DIR"], f"{optimizer_name}_chromosomes.json"), "w"),
        indent=2,
    )
    json.dump(
        list(logs),
        open(os.path.join(CONFIG["OUTPUT_DIR"], f"{optimizer_name}_logs.json"), "w"),
        indent=2,
    )

    if isinstance(logs[0], dict) and "fitness_curve" in logs[0]:
        plt.figure()
        for i, log in enumerate(logs):
            plt.plot(log["fitness_curve"], label=f"Trial {i+1}")
        plt.title(f"{optimizer_name} Fitness over Generations (Reference Scenario)")
        plt.xlabel("Generation or Episode")
        plt.ylabel("Fitness on Reference Scenario")
        plt.legend()
        plt.savefig(os.path.join(CONFIG["OUTPUT_DIR"], f"{optimizer_name}_fitness.png"))
        plt.close()


def final_comparison(best_chromosomes):
    # Use a shared fixed scenario
    scenario = get_final_comparison_scenario()
    results = {}
    for optimizer, chromosome in best_chromosomes.items():
        score = evaluate_chromosome(chromosome, scenario)
        results[optimizer] = score

    plt.figure()
    plt.bar(results.keys(), results.values())
    plt.title("Final Optimizer Comparison on Shared Scenario")
    plt.ylabel("Final Score")
    plt.savefig(os.path.join(CONFIG["OUTPUT_DIR"], "final_comparison.png"))
    plt.close()
    return results


def overlay_best_curves(all_logs):
    """
    Plot best trial fitness curves vs raw generation index
    using the reference scenario fitness.
    """
    plt.figure()
    for opt, logs in all_logs.items():
        best_log = max(logs, key=lambda l: l["fitness_curve"][-1])
        plt.plot(best_log["fitness_curve"], label=opt)
    plt.title("Best Trial Fitness Comparison (Reference Scenario)")
    plt.xlabel("Generation or Episode")
    plt.ylabel("Fitness on Reference Scenario")
    plt.legend()
    plt.savefig(os.path.join(CONFIG["OUTPUT_DIR"], "optimizer_overlay.png"))
    plt.close()


def overlay_best_curves_percent(all_logs):
    """
    Plot best trial fitness curves vs percent of generations
    or episodes completed so that curves of different lengths
    align from 0 to 100 percent.
    """
    plt.figure()
    for opt, logs in all_logs.items():
        best_log = max(logs, key=lambda l: l["fitness_curve"][-1])
        curve = best_log["fitness_curve"]
        if len(curve) <= 1:
            x_vals = [0.0] * len(curve)
        else:
            x_vals = [100.0 * i / (len(curve) - 1) for i in range(len(curve))]
        plt.plot(x_vals, curve, label=opt)
    plt.title("Best Trial Fitness vs Generations Percent (Reference Scenario)")
    plt.xlabel("Generations Percent")
    plt.ylabel("Fitness on Reference Scenario")
    plt.legend()
    plt.savefig(os.path.join(CONFIG["OUTPUT_DIR"], "optimizer_overlay_percent.png"))
    plt.close()


def run_all():
    np.random.seed(CONFIG["SEED"])
    best_chromosomes = {}
    all_logs = {}

    for optimizer in CONFIG["OPTIMIZERS"]:
        print(f"[{datetime.now()}] === Running optimizer: {optimizer} ===")
        scores, chromosomes, logs = run_optimizer_trials(optimizer)
        scores, chromosomes, logs = list(scores), list(chromosomes), list(logs)
        save_results(optimizer, scores, chromosomes, logs)
        best_idx = np.argmax(scores)
        best_chromosomes[optimizer] = chromosomes[best_idx]
        all_logs[optimizer] = logs

    final_results = final_comparison(best_chromosomes)
    overlay_best_curves(all_logs)
    overlay_best_curves_percent(all_logs)
    print(f"[{datetime.now()}] All evaluations complete. Final results: {final_results}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_all()
