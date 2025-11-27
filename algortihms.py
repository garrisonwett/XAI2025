import os
import sys
import time
import random
import argparse
import multiprocessing
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from kesslergame import TrainerEnvironment, GraphicsType, KesslerGame
from kesslergame.scenario import Scenario

# Make sure we can import your controller
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from redone_controller import FuzzyController

# Import your scenarios and frozen map helper
from scenarios import scenarios as SCENARIOS, random_repeatable_frozen

# -------------------------------------------------------------------
# Default GA hyperparameters and globals
# -------------------------------------------------------------------

DEFAULT_CHROMOSOME_LENGTH = 68         # Length of chromosome
DEFAULT_POPULATION_SIZE = 40            # Number of individuals
DEFAULT_MAX_GENERATIONS = 200           # Number of generations
DEFAULT_CROSSOVER_RATE = 0.8            # Probability of crossover

# Mutation schedule start and end over generations
DEFAULT_MUTATION_RATE_START = 0.9       # Early exploration
DEFAULT_MUTATION_RATE_END = 0.05        # Late exploitation
DEFAULT_MUTATION_STEP_START = 0.9
DEFAULT_MUTATION_STEP_END = 0.02

DEFAULT_TOURNAMENT_K = 3                # Tournament size
DEFAULT_NUM_CORES = max(1, multiprocessing.cpu_count() - 4)

DEFAULT_DEATH_PENALTY_SCALE = 10.0      # Kept for compatibility (not used in new fitness)

# Time limit in hours. None means no limit.
DEFAULT_MAX_HOURS = None

RESULTS_ROOT = "Results"

# When no explicit training maps are provided, this many random frozen maps
# are generated per generation using random_repeatable_frozen
NUM_RANDOM_TRAINING_MAPS = 10

# Stagnation handling
STAGNATION_PATIENCE = 20            # generations without improvement
STAGNATION_BOOST_FACTOR = 1.5
STAGNATION_MAX_MUT_RATE = 0.7
STAGNATION_MAX_MUT_STEP = 0.3

# Elitism and immigrants
ELITE_FRACTION = 0.1                # keep top 10 percent as elites
MIN_ELITES = 2                      # at least this many elites
IMMIGRANT_FRACTION = 0.05           # fraction of population replaced by random individuals when stagnant

# Gene discretization
GENE_RESOLUTION = 0.001             # round genes to nearest 0.001

# KesslerGame and TrainerEnvironment settings
# TrainerEnvironment will ignore graphics, KesslerGame will show them
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

# Lazily created per process
_GAME_ENV = None

# -------------------------------------------------------------------
# GA building blocks
# -------------------------------------------------------------------

def get_game_env() -> TrainerEnvironment:
    """
    Lazily create a TrainerEnvironment per process.
    """
    global _GAME_ENV
    if _GAME_ENV is None:
        _GAME_ENV = TrainerEnvironment(settings=game_settings)
    return _GAME_ENV


def create_random_individual(length: int) -> np.ndarray:
    """
    Create a random chromosome of a given length with genes in [0, 1],
    then snap to a discrete grid if GENE_RESOLUTION is set.
    """
    ind = np.random.rand(length).astype(float)
    if GENE_RESOLUTION is not None and GENE_RESOLUTION > 0:
        ind = np.round(ind / GENE_RESOLUTION) * GENE_RESOLUTION
        ind = np.clip(ind, 0.0, 1.0)
    return ind


def tournament_selection_index(population: List[np.ndarray], fitnesses: List[float], k: int) -> int:
    """
    Select one parent index via tournament selection.
    """
    best_idx = None
    best_fit = -np.inf
    n = len(population)
    for _ in range(k):
        idx = random.randrange(n)
        fit = fitnesses[idx]
        if fit > best_fit or best_idx is None:
            best_fit = fit
            best_idx = idx
    return int(best_idx)


def tournament_selection(population: List[np.ndarray], fitnesses: List[float], k: int) -> np.ndarray:
    """
    Select one parent via tournament selection, returning a copy of the chromosome.
    """
    idx = tournament_selection_index(population, fitnesses, k)
    return population[idx].copy()


def crossover(parent1: np.ndarray, parent2: np.ndarray, rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover with probability rate.
    """
    if random.random() < rate:
        mask = np.random.rand(parent1.size) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
    else:
        child1 = parent1.copy()
        child2 = parent2.copy()
    return child1, child2


def mutate(ind: np.ndarray, rate: float, step: float) -> np.ndarray:
    """
    Per gene mutation.
    Adds uniform noise in [-step, step] to each selected gene,
    clamps back to [0, 1], and optionally snaps to a discrete grid.
    """
    mask = np.random.rand(ind.size) < rate
    if not np.any(mask):
        return ind

    noise = np.random.uniform(-step, step, size=ind.size)
    mutated = ind + noise
    mutated = np.clip(mutated, 0.0, 1.0)
    out = np.where(mask, mutated, ind)

    if GENE_RESOLUTION is not None and GENE_RESOLUTION > 0:
        out = np.round(out / GENE_RESOLUTION) * GENE_RESOLUTION
        out = np.clip(out, 0.0, 1.0)

    return out


def fitness_on_scenario(
    env: TrainerEnvironment,
    chromosome: np.ndarray,
    scenario: Scenario,
    death_penalty_scale: float,
) -> float:
    """
    Run one scenario and compute a richer fitness.

    Fitness components:
      - Normalized kills
      - Accuracy
      - Survival time
      - Penalty for deaths

    death_penalty_scale is kept for compatibility but is not used
    directly in this new fitness.
    """
    score, _ = env.run(
        chromosome,
        scenario=scenario,
        controllers=[FuzzyController()],
    )

    team = score.teams[0]

    asteroids_hit = float(team.asteroids_hit)
    deaths = float(team.deaths)
    accuracy = float(getattr(team, "accuracy", 0.0))

    num_asteroids = float(len(scenario.asteroid_states)) if scenario.asteroid_states is not None else float(
        getattr(scenario, "num_asteroids", 0) or 0
    )
    if num_asteroids <= 0:
        num_asteroids = 1.0
    kills_norm = asteroids_hit / num_asteroids

    time_limit = float(getattr(scenario, "time_limit", 60.0))
    time_alive = float(getattr(team, "time_alive", time_limit))
    time_norm = time_alive / max(1.0, time_limit)

    deaths_capped = min(deaths, 3.0) / 3.0

    # Weighted sum
    fitness = (
        3.0 * kills_norm +
        1.5 * accuracy +
        1.0 * time_norm -
        4.0 * deaths_capped
    )

    # Clip to avoid extreme negative outliers dominating the GA
    fitness = max(fitness, -50.0)

    return fitness


def fitness_function(
    chromosome: np.ndarray,
    training_scenarios: List[Scenario],
    death_penalty_scale: float = DEFAULT_DEATH_PENALTY_SCALE,
) -> float:
    """
    Fitness for a single chromosome.

    Uses a bundle of training scenarios.
    Fitness is the average over all training scenarios in the provided list.
    """
    env = get_game_env()
    total = 0.0

    for scenario in training_scenarios:
        total += fitness_on_scenario(env, chromosome, scenario, death_penalty_scale)

    return total / float(len(training_scenarios))


# Helper for starmap
def _fitness_wrapper(args):
    chromosome, training_scenarios, death_penalty_scale = args
    return fitness_function(chromosome, training_scenarios, death_penalty_scale)

# -------------------------------------------------------------------
# Main GA loop
# -------------------------------------------------------------------

def genetic_algorithm(
    chromosome_length: int = DEFAULT_CHROMOSOME_LENGTH,
    population_size: int = DEFAULT_POPULATION_SIZE,
    max_generations: int = DEFAULT_MAX_GENERATIONS,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate_start: float = DEFAULT_MUTATION_RATE_START,
    mutation_rate_end: float = DEFAULT_MUTATION_RATE_END,
    mutation_step_start: float = DEFAULT_MUTATION_STEP_START,
    mutation_step_end: float = DEFAULT_MUTATION_STEP_END,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
    num_cores: int = DEFAULT_NUM_CORES,
    death_penalty_scale: float = DEFAULT_DEATH_PENALTY_SCALE,
    max_hours: Optional[float] = DEFAULT_MAX_HOURS,
    training_maps: Optional[List[Scenario]] = None,
) -> Tuple[np.ndarray, float, List[float], str]:
    """
    Run the genetic algorithm.

    If training_maps is provided and non empty, those maps are used for every generation.
    If training_maps is None or empty, each generation uses NUM_RANDOM_TRAINING_MAPS
    maps produced by random_repeatable_frozen for that generation index.

    Returns:
        best_chromosome
        best_training_fitness
        training_best_history   best fitness per generation
        run_dir                 path to Results/GA_[timestamp] folder
    """
    # Normalize training_maps
    if training_maps is not None and len(training_maps) == 0:
        training_maps = None

    # Prepare run directory
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"GA_{timestamp}"
    run_dir = os.path.join(RESULTS_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Initial population
    population = [create_random_individual(chromosome_length) for _ in range(population_size)]

    best_overall = None
    best_overall_fit = -np.inf
    training_best_history: List[float] = []

    # Extra diagnostics
    pop_avg_history: List[float] = []
    pop_median_history: List[float] = []
    pop_min_history: List[float] = []
    pop_max_history: List[float] = []

    fitness_age = 0  # stagnation counter

    start_time = time.perf_counter()
    num_cores = max(1, int(num_cores))

    with multiprocessing.Pool(processes=num_cores) as pool:
        for gen in range(max_generations):
            # Time based stopping
            if max_hours is not None:
                elapsed_hours = (time.perf_counter() - start_time) / 3600.0
                if elapsed_hours >= max_hours:
                    print(
                        f"\nReached max time limit of {max_hours:.2f} hours "
                        f"at generation {gen}. Stopping GA."
                    )
                    break

            gen_start = time.perf_counter()

            # Mutation schedule based on progress
            if max_generations > 1:
                progress = gen / float(max_generations - 1)
            else:
                progress = 1.0

            base_mutation_rate = mutation_rate_start + progress * (mutation_rate_end - mutation_rate_start)
            base_mutation_step = mutation_step_start + progress * (mutation_step_end - mutation_step_start)

            # Choose training scenarios for this generation
            if training_maps is not None:
                scenarios_for_this_gen = training_maps
            else:
                scenarios_for_this_gen = []
                for i in range(NUM_RANDOM_TRAINING_MAPS):
                    scn = random_repeatable_frozen(
                        gen_idx=gen,
                        map_idx=i,
                    )
                    scenarios_for_this_gen.append(scn)

            # Evaluate fitness in parallel on training scenarios
            args_list = [
                (ind, scenarios_for_this_gen, death_penalty_scale)
                for ind in population
            ]
            fitnesses = pool.map(_fitness_wrapper, args_list)

            # Generation statistics
            gen_best_fit = max(fitnesses)
            gen_best_idx = fitnesses.index(gen_best_fit)
            gen_best_ind = population[gen_best_idx].copy()

            avg_fit = float(np.mean(fitnesses))
            med_fit = float(np.median(fitnesses))
            min_fit = float(np.min(fitnesses))
            max_fit_val = float(np.max(fitnesses))

            pop_avg_history.append(avg_fit)
            pop_median_history.append(med_fit)
            pop_min_history.append(min_fit)
            pop_max_history.append(max_fit_val)

            # Track best overall and stagnation
            if gen_best_fit > best_overall_fit + 1e-9:
                best_overall_fit = gen_best_fit
                best_overall = gen_best_ind.copy()
                fitness_age = 0
            else:
                fitness_age += 1

            # Start from scheduled mutation parameters
            mutation_rate = base_mutation_rate
            mutation_step = base_mutation_step

            # Stagnation boost if no improvement for a while
            if fitness_age >= STAGNATION_PATIENCE:
                mutation_rate = min(mutation_rate * STAGNATION_BOOST_FACTOR, STAGNATION_MAX_MUT_RATE)
                mutation_step = min(mutation_step * STAGNATION_BOOST_FACTOR, STAGNATION_MAX_MUT_STEP)
                print(
                    f"Stagnation detected (age {fitness_age}). "
                    f"Boosting mutation_rate to {mutation_rate:.3f}, "
                    f"mutation_step to {mutation_step:.3f}"
                )

            training_best_history.append(gen_best_fit)

            gen_time = time.perf_counter() - gen_start

            # Printout at end of generation
            print(
                f"[Gen {gen}] time: {gen_time:.2f} sec  "
                f"mut_rate_base: {base_mutation_rate:.3f}  mut_step_base: {base_mutation_step:.3f}  "
                f"train_best: {gen_best_fit:.3f}  "
                f"pop_avg: {avg_fit:.3f}  pop_med: {med_fit:.3f}  "
                f"pop_min: {min_fit:.3f}  pop_max: {max_fit_val:.3f}  "
                f"overall_best_train: {best_overall_fit:.3f}"
            )

            # Elitism
            elite_count = max(MIN_ELITES, int(population_size * ELITE_FRACTION))
            elite_count = min(elite_count, population_size)
            sorted_idx = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)
            elites = [population[i].copy() for i in sorted_idx[:elite_count]]

            new_population: List[np.ndarray] = elites.copy()

            # Adaptive mutation based on parent fitness relative to median
            median_fit = float(np.median(fitnesses))

            # Create rest of next population
            while len(new_population) < population_size:
                p1_idx = tournament_selection_index(population, fitnesses, tournament_k)
                p2_idx = tournament_selection_index(population, fitnesses, tournament_k)
                p1 = population[p1_idx]
                p2 = population[p2_idx]

                c1, c2 = crossover(p1, p2, crossover_rate)

                rate1 = mutation_rate * (1.5 if fitnesses[p1_idx] < median_fit else 0.7)
                rate2 = mutation_rate * (1.5 if fitnesses[p2_idx] < median_fit else 0.7)

                c1 = mutate(c1, rate1, mutation_step)
                c2 = mutate(c2, rate2, mutation_step)

                new_population.append(c1)
                if len(new_population) < population_size:
                    new_population.append(c2)

            # Random immigrants when stuck to reintroduce diversity
            if fitness_age >= STAGNATION_PATIENCE:
                num_immigrants = max(1, int(IMMIGRANT_FRACTION * population_size))
                for i in range(num_immigrants):
                    replace_idx = population_size - 1 - i
                    if replace_idx <= 0:
                        break
                    new_population[replace_idx] = create_random_individual(chromosome_length)

            population = new_population

    total_time = time.perf_counter() - start_time
    print(f"\nGA finished in {total_time:.2f} seconds")
    print(f"Best overall training fitness: {best_overall_fit:.3f}")
    print(f"Best chromosome:\n{best_overall}")

    # Convert lists to arrays
    training_best_arr = np.array(training_best_history, dtype=float)
    pop_avg_arr = np.array(pop_avg_history, dtype=float)
    pop_med_arr = np.array(pop_median_history, dtype=float)
    pop_min_arr = np.array(pop_min_history, dtype=float)
    pop_max_arr = np.array(pop_max_history, dtype=float)
    gens = np.arange(len(training_best_arr))

    # Save best chromosome as text
    best_chr_path = os.path.join(run_dir, "best_chromsome.txt")
    if best_overall is not None:
        np.savetxt(best_chr_path, best_overall, fmt="%.6f")
    else:
        with open(best_chr_path, "w") as f:
            f.write("No generations were completed, no best chromosome available.\n")

    # Plot fitness vs generations and save
    if len(training_best_arr) > 0:
        fig, ax = plt.subplots()
        ax.plot(gens, training_best_arr, label="Training best")
        ax.plot(gens, pop_avg_arr, label="Population average")
        ax.plot(gens, pop_med_arr, label="Population median")
        ax.plot(gens, pop_min_arr, label="Population min")
        ax.plot(gens, pop_max_arr, label="Population max")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Genetic Algorithm Progress")
        ax.legend()
        fig.tight_layout()
        plot_path = os.path.join(run_dir, "fitness_vs_generation.png")
        fig.savefig(plot_path)
        plt.close(fig)
    else:
        plot_path = os.path.join(run_dir, "fitness_vs_generation.png")
        with open(plot_path, "w") as f:
            f.write("No generations were completed, no plot available.\n")

    # Save parameters and all scores into one text file
    params_path = os.path.join(run_dir, "ga_parameters.txt")
    with open(params_path, "w") as f:
        f.write("Genetic Algorithm Run Parameters and Results\n")
        f.write(f"Run directory: {run_dir}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("Parameters:\n")
        f.write(f"  chromosome_length         = {chromosome_length}\n")
        f.write(f"  population_size           = {population_size}\n")
        f.write(f"  max_generations           = {max_generations}\n")
        f.write(f"  crossover_rate            = {crossover_rate}\n")
        f.write(f"  mutation_rate_start       = {mutation_rate_start}\n")
        f.write(f"  mutation_rate_end         = {mutation_rate_end}\n")
        f.write(f"  mutation_step_start       = {mutation_step_start}\n")
        f.write(f"  mutation_step_end         = {mutation_step_end}\n")
        f.write(f"  tournament_k              = {tournament_k}\n")
        f.write(f"  num_cores                 = {num_cores}\n")
        f.write(f"  death_penalty_scale       = {death_penalty_scale}\n")
        if training_maps is not None:
            f.write(f"  training_mode             = fixed_maps\n")
            f.write(f"  num_training_maps         = {len(training_maps)}\n")
            f.write(f"  training_map_names        = {[s.name for s in training_maps]}\n")
        else:
            f.write(f"  training_mode             = random_frozen\n")
            f.write(f"  num_training_maps         = {NUM_RANDOM_TRAINING_MAPS}\n")
        f.write(f"  max_hours                 = {max_hours}\n")
        f.write(f"  gene_resolution           = {GENE_RESOLUTION}\n")
        f.write(f"  stagnation_patience       = {STAGNATION_PATIENCE}\n")
        f.write(f"  stagnation_boost_factor   = {STAGNATION_BOOST_FACTOR}\n")
        f.write(f"  stagnation_max_mut_rate   = {STAGNATION_MAX_MUT_RATE}\n")
        f.write(f"  stagnation_max_mut_step   = {STAGNATION_MAX_MUT_STEP}\n")
        f.write(f"  elite_fraction            = {ELITE_FRACTION}\n")
        f.write(f"  min_elites                = {MIN_ELITES}\n")
        f.write(f"  immigrant_fraction        = {IMMIGRANT_FRACTION}\n\n")

        f.write("Summary:\n")
        f.write(f"  total_time_seconds        = {total_time:.4f}\n")
        f.write(f"  best_overall_training     = {best_overall_fit:.6f}\n")
        f.write(f"  num_generations_ran       = {len(training_best_arr)}\n\n")

        f.write("Per generation scores:\n")
        f.write("  gen_index, training_best, pop_avg, pop_median, pop_min, pop_max\n")
        for g, tr, pa, pm, pmin, pmax in zip(
            gens, training_best_arr, pop_avg_arr, pop_med_arr, pop_min_arr, pop_max_arr
        ):
            f.write(
                f"  {int(g)}, {tr:.6f}, {pa:.6f}, {pm:.6f}, {pmin:.6f}, {pmax:.6f}\n"
            )

    return best_overall, best_overall_fit, training_best_history, run_dir

# -------------------------------------------------------------------
# Visual demo helper
# -------------------------------------------------------------------

def run_best_chromosome_visual(best_chromosome: np.ndarray, scenario: Scenario) -> None:
    """
    Run the best chromosome in a visual KesslerGame on a given scenario.
    """
    demo_settings = dict(game_settings)
    demo_settings["prints_on"] = True
    demo_settings["UI_settings"] = {
        "ships": True,
        "lives_remaining": True,
        "accuracy": True,
        "asteroids_hit": True,
        "shots_fired": True,
        "bullets_remaining": True,
        "controller_name": True,
    }
    demo_settings["graphics_type"] = GraphicsType.Tkinter

    game = KesslerGame(settings=demo_settings)

    print(f"\nLaunching visual run on scenario '{scenario.name}'...")
    score, perf_data = game.run(
        best_chromosome,
        scenario=scenario,
        controllers=[FuzzyController()],
    )
    print("Visual run finished.")
    team = score.teams[0]
    print(f"Asteroids hit: {team.asteroids_hit}, deaths: {team.deaths}, accuracy: {team.accuracy:.3f}")

# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GA to optimize fuzzy controller for Asteroids")

    parser.add_argument("--chromosome_length", type=int, default=DEFAULT_CHROMOSOME_LENGTH,
                        help="Length of chromosome")
    parser.add_argument("--generations", type=int, default=DEFAULT_MAX_GENERATIONS,
                        help="Number of generations")
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION_SIZE,
                        help="Population size")
    parser.add_argument("--cores", type=int, default=DEFAULT_NUM_CORES,
                        help="Worker processes for fitness evaluation")
    parser.add_argument("--crossover", type=float, default=DEFAULT_CROSSOVER_RATE,
                        help="Crossover rate")
    parser.add_argument("--mutation_rate_start", type=float, default=DEFAULT_MUTATION_RATE_START,
                        help="Initial per gene mutation rate")
    parser.add_argument("--mutation_rate_end", type=float, default=DEFAULT_MUTATION_RATE_END,
                        help="Final per gene mutation rate")
    parser.add_argument("--mutation_step_start", type=float, default=DEFAULT_MUTATION_STEP_START,
                        help="Initial mutation step size")
    parser.add_argument("--mutation_step_end", type=float, default=DEFAULT_MUTATION_STEP_END,
                        help="Final mutation step size")
    parser.add_argument("--tournament_k", type=int, default=DEFAULT_TOURNAMENT_K,
                        help="Tournament size for parent selection")
    parser.add_argument("--death_penalty_scale", type=float, default=DEFAULT_DEATH_PENALTY_SCALE,
                        help="Kept for compatibility (not used directly in new fitness)")
    parser.add_argument("--max_hours", type=float, default=DEFAULT_MAX_HOURS,
                        help="Max run time in hours. If set, no new generation starts after this limit.")

    args = parser.parse_args()

    # For now, use your existing training scenarios
    training_maps = [
        SCENARIOS["training1"],
        SCENARIOS["training2"],
    ]

    best_chromosome, best_fitness, train_hist, run_dir = genetic_algorithm(
        chromosome_length=args.chromosome_length,
        population_size=args.population,
        max_generations=args.generations,
        crossover_rate=args.crossover,
        mutation_rate_start=args.mutation_rate_start,
        mutation_rate_end=args.mutation_rate_end,
        mutation_step_start=args.mutation_step_start,
        mutation_step_end=args.mutation_step_end,
        tournament_k=args.tournament_k,
        num_cores=args.cores,
        death_penalty_scale=args.death_penalty_scale,
        max_hours=args.max_hours,
        training_maps=training_maps,
    )

    print(f"\nRun results saved in: {run_dir}")

    # Final interactive prompt
    try:
        choice = input('Press "e" to display best chromsome or press any other button to end: ').strip().lower()
    except EOFError:
        choice = ""

    if choice == "e" and best_chromosome is not None:
        # Show visual run on training1 by default
        vis_scenario = SCENARIOS["training1"]
        run_best_chromosome_visual(best_chromosome, vis_scenario)
    else:
        print("Exiting without visual run.")

if __name__ == "__main__":
    main()
