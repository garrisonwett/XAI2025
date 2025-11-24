import random
import argparse
import os
import sys
import time
from datetime import datetime
import traceback
import numpy as np
from kesslergame import GraphicsType, KesslerGame, Scenario, TrainerEnvironment
import multiprocessing
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redone_controller import FuzzyController
from scenarios import scenarios
from utils import LoggerUtility, LoggingLevel

# Set up the logger
logger = LoggerUtility(LoggingLevel.DEBUG).get_logger()

# ----------------------------
# Global debug and safety settings
# ----------------------------
# Maximum wall clock time that a single fitness evaluation is "allowed"
# before we log a loud warning. This does not kill it, but it tells you
# exactly which individual is causing the slowdown.
INDIVIDUAL_WARN_SECONDS = 60.0

# In game time limit for each simulation to avoid endless episodes.
# You can tune this, but avoid float("inf") for debugging.
GAME_TIME_LIMIT = 60.0

# Small epsilon to keep genes strictly within (0, 1)
EPSILON = 1e-6

# All Available Settings
game_settings = {
    "frequency": 30,  # Dictates both frequency and time_step settings (float)
    "perf_tracker": True,  # (bool)
    "prints_on": True,  # (bool)
    "graphics_type": GraphicsType.Tkinter,  # (GraphicsType)
    "graphics_obj": None,  # (Optional[KesslerGraphics])
    "realtime_multiplier": 1,  # (float)
    "time_limit": GAME_TIME_LIMIT,  # (float) was float("inf")
    "random_ast_splits": False,  # (bool)
    "UI_settings": {
        "ships": True,  # (bool)
        "lives_remaining": True,  # (bool)
        "accuracy": True,  # (bool)
        "asteroids_hit": True,  # (bool)
        "shots_fired": True,  # (bool)
        "bullets_remaining": True,  # (bool)
        "controller_name": True,  # (bool)
    },  # (Dict[str, bool])
}

# ----------------------------
# 1. Hyperparameters
# ----------------------------
CHROMOSOME_SIZE     = 177      # Number of genes per individual
POPULATION_SIZE     = 80       # How many individuals in each generation
MAX_GENERATIONS     = 3000     # Maximum number of GA iterations
MUTATION_RATE_BASE  = 0.5      # Starting mutation probability per gene
CROSSOVER_RATE_BASE = 0.8      # Starting crossover probability per mating
CROSSOVER_INCREASE  = 0.9      # Final (max) crossover probability
TOURNAMENT_K        = 3        # Tournament size for parent selection
POOL_PROCESSES      = max(1, multiprocessing.cpu_count() - 4)  # Number of worker processes for fitness eval
USE_MULTIPROCESSING = True    # Toggle parallel fitness evaluation on or off
STOP_HOURS          = 12       # Maximum hours to run the GA

# ----------------------------
# 2. Prepare simulators once
# ----------------------------
scenario_names = ["training2"]
# Use fast TrainerEnvironment for fitness; only KesslerGame for final demo
games = {
    name: TrainerEnvironment(settings=game_settings)
    for name in scenario_names
}

# ----------------------------
# 3. Helper functions
# ----------------------------

def create_random_individual(size):
    """
    Create a random chromosome of given length with gene values in (0, 1).

    We use np.random.random in [0, 1) and then clip to [EPSILON, 1 - EPSILON]
    so that no gene is ever exactly 0 or 1.
    """
    vals = np.random.random(size).astype(float)
    vals = np.clip(vals, EPSILON, 1.0 - EPSILON)
    return vals


def fitness_function(chromosome):
    """
    Compute the fitness of one individual by simulating each scenario.

    Uses pre-instantiated TrainerEnvironment instances to avoid
    rebuilding parsers or environments on every call, which greatly
    reduced per-evaluation overhead.
    
    Returns the sum over scenarios of (asteroids_hit - 30 * deaths^3 - time penalty).
    """
    total = 0.0
    try:
        for name in scenario_names:
            logger.debug("Starting game.run for scenario %s", name)
            print(f"[fitness_function] Starting game.run for scenario {name}", flush=True)

            score, _ = games[name].run(
                chromosome,
                scenario=scenarios[name],
                controllers=[FuzzyController()]
            )

            logger.debug("Finished game.run for scenario %s", name)
            team = score.teams[0]
            msg = (
                f"Scenario: {name}, Asteroids Hit: {team.asteroids_hit}, "
                f"Deaths: {team.deaths}, Accuracy: {team.accuracy:.4f}, "
                f"Mean Eval Time: {team.mean_eval_time:.4f}s"
            )
            print(msg, flush=True)
            logger.debug(msg)

            total += (team.accuracy * team.asteroids_hit) - 30 * (team.deaths**3) - team.mean_eval_time * 100
    except Exception as e:
        # Log full traceback so we know exactly where it failed
        logger.error("Exception inside fitness_function: %s", str(e))
        traceback.print_exc()
        raise
    return total


def guarded_fitness_evaluation(idx, chromosome):
    """
    Wrapper around fitness_function that:
      - Logs which individual is being evaluated.
      - Times the evaluation and warns if it is slow.
      - Catches and logs any exception with index and chromosome snippet.
    """
    start_time = time.perf_counter()
    print(f"[GA] Starting fitness eval for individual {idx}", flush=True)
    logger.debug("Starting fitness eval for individual %d", idx)

    try:
        fitness = fitness_function(chromosome)
    except Exception as e:
        logger.error("Error during fitness evaluation for individual %d: %s", idx, str(e))
        traceback.print_exc()
        # Optionally dump part of the chromosome for debugging
        chrom_preview = ", ".join(f"{g:.3f}" for g in chromosome[:10])
        print(f"[GA] Failed individual index {idx}, first 10 genes: [{chrom_preview}]",
              flush=True)
        raise

    elapsed = time.perf_counter() - start_time
    logger.debug("Finished fitness eval for individual %d in %.3f seconds (fitness=%.6f)",
                 idx, elapsed, fitness)
    print(
        f"[GA] Finished fitness eval for individual {idx} in {elapsed:.2f} seconds, "
        f"fitness={fitness:.6f}",
        flush=True
    )

    if elapsed > INDIVIDUAL_WARN_SECONDS:
        warn_msg = (
            f"[GA] WARNING: Fitness eval for individual {idx} took {elapsed:.2f} seconds "
            f"(over {INDIVIDUAL_WARN_SECONDS} s). Likely culprit for perceived freeze."
        )
        print(warn_msg, flush=True)
        logger.warning(warn_msg)

    return fitness


def tournament_selection(population, fitnesses, k):
    """
    Select one parent via tournament selection.
    
    Randomly samples k individuals (with replacement) and returns
    the one with the highest fitness. This balances exploration
    (random sampling) and exploitation (picking the best).
    """
    best_ind, best_fit = None, -np.inf
    for _ in range(k):
        idx = random.randrange(len(population))
        fit = fitnesses[idx]
        if fit > best_fit:
            best_fit = fit
            best_ind = population[idx]
    return best_ind


def crossover(parent1, parent2, rate):
    """
    With probability `rate`, perform one-point crossover between two parents.
    
    A random cut point is chosen (excluding ends), and genes are
    spliced to create two children.
    Otherwise, children are exact copies of their parents.
    """
    if random.random() < rate:
        cp = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:cp], parent2[cp:]))
        child2 = np.concatenate((parent2[:cp], parent1[cp:]))
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    return child1, child2


def mutate(parent: np.ndarray, rate: float, distance: float) -> np.ndarray:
    """
    Perform per-gene mutation with probability `rate`, where each mutation 
    shifts the gene by at most `distance` but always stays within (0, 1).

    The resulting genes are clipped to [EPSILON, 1 - EPSILON] to prevent
    any value from being exactly 0 or 1.
    """
    # 1) decide which genes to mutate
    mask = np.random.rand(parent.size) < rate

    # 2) for each gene, compute its allowable noise interval [low, high]
    #    so that parent[i] + noise remains in [0,1]
    low  = np.maximum(-distance,      -parent)
    high = np.minimum( distance, 1.0 - parent)

    # 3) sample noise uniformly in [low, high] for every gene
    u = np.random.rand(parent.size)
    noise = low + u * (high - low)

    # 4) apply noise and clip to keep everything strictly inside (0, 1)
    mutated = parent + noise
    mutated = np.clip(mutated, EPSILON, 1.0 - EPSILON)

    # 5) only apply to genes selected by the mask
    return np.where(mask, mutated, parent)


# ----------------------------
# 4. Main GA loop
# ----------------------------
def genetic_algorithm():
    """
    Runs the GA with dynamic rates, elitism, stagnation boost,
    optional parallel fitness evaluation, and a time-based stop criterion.
    """
    # 1) Initialize population
    population = [create_random_individual(CHROMOSOME_SIZE)
                  for _ in range(POPULATION_SIZE)]
    best_solution_ever = None
    best_fitness_ever = -np.inf
    last_fitness = None
    fitness_age = 0
    fitness_tracker = []

    # 2) Keep track of start time
    ga_start_time = time.perf_counter()

    # 3) Optionally create a multiprocessing pool
    if USE_MULTIPROCESSING and POOL_PROCESSES > 1:
        print(f"Using multiprocessing with {POOL_PROCESSES} worker processes.")
        logger.info("Using multiprocessing with %d worker processes.", POOL_PROCESSES)
        pool = multiprocessing.Pool(POOL_PROCESSES)
    else:
        print("Running fitness evaluations in a single process (no multiprocessing).")
        logger.info("Running fitness evaluations in a single process (no multiprocessing).")
        pool = None

    for generation in range(MAX_GENERATIONS):
        # Check overall time limit
        elapsed_hours = (time.perf_counter() - ga_start_time) / 3600.0
        if elapsed_hours >= STOP_HOURS:
            print(f"Reached time limit of {STOP_HOURS} hours. Stopping GA at generation {generation}.")
            logger.warning("Reached GA time limit of %d hours at generation %d.",
                           STOP_HOURS, generation)
            break

        gen_start = time.perf_counter()
        print(f"========== Generation {generation} ==========", flush=True)
        logger.info("Starting generation %d", generation)

        # 4) Update dynamic rates
        mutation_rate = MUTATION_RATE_BASE * (1 - generation / MAX_GENERATIONS)
        crossover_rate = (CROSSOVER_RATE_BASE +
                          (CROSSOVER_INCREASE - CROSSOVER_RATE_BASE) *
                          (generation / 300))
        
        crossover_rate = min(crossover_rate, CROSSOVER_INCREASE)

        if fitness_age >= 20:
            boosted = MUTATION_RATE_BASE + (1 - MUTATION_RATE_BASE) * (fitness_age / 100)
            mutation_rate = min(boosted, 0.7)
            print(f"Boosted mutation rate to {mutation_rate:.4f}")
            logger.info("Boosted mutation rate to %.4f due to stagnation.", mutation_rate)
        
        print(f"Mutation rate: {mutation_rate:.4f}, "
              f"Crossover rate: {crossover_rate:.4f}", flush=True)
        logger.debug("Mutation rate: %.4f, Crossover rate: %.4f",
                     mutation_rate, crossover_rate)
    
        # 5) Evaluate fitnesses (parallel or single process)
        if pool is not None:
            # For debugging, you can switch to guarded_fitness_evaluation in multiprocessing
            # using pool.starmap, but interleaved logging can be noisy.
            fitnesses = pool.map(fitness_function, population)
        else:
            fitnesses = [
                guarded_fitness_evaluation(i, ind)
                for i, ind in enumerate(population)
            ]

        # 6) Track best solution
        current_best_fit = max(fitnesses)
        current_best_ind = population[fitnesses.index(current_best_fit)]
        if current_best_fit > best_fitness_ever:
            best_fitness_ever = current_best_fit
            best_solution_ever = current_best_ind.copy()

        # 7) Logging
        print(f"Generation {generation}, Best Fitness this generation {current_best_fit:.6f}, "
              f"Best Fitness so far: {best_fitness_ever:.6f}", flush=True)
        logger.info(
            "Generation %d complete. Best fitness this gen: %.6f, best ever: %.6f",
            generation, current_best_fit, best_fitness_ever
        )

        fitness_tracker.append(best_fitness_ever)

        # 8) Elitism and stagnation boost
        new_population = [current_best_ind.copy()]
        if current_best_fit == last_fitness:
            fitness_age += 1
        else:
            fitness_age = 0

        # 9) Generate next population
        distance = (1 - generation) / MAX_GENERATIONS
        while len(new_population) < POPULATION_SIZE:
            p1 = tournament_selection(population, fitnesses, TOURNAMENT_K)
            p2 = tournament_selection(population, fitnesses, TOURNAMENT_K)
            c1, c2 = crossover(p1, p2, crossover_rate)
            new_population.append(mutate(c1, mutation_rate, distance))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(c2, mutation_rate, distance))

        population = new_population[:POPULATION_SIZE]
        last_fitness = current_best_fit

        gen_elapsed = time.perf_counter() - gen_start
        print(f"Generation {generation} completed in {gen_elapsed:.2f} seconds.\n", flush=True)
        logger.info("Generation %d completed in %.2f seconds.", generation, gen_elapsed)

        print(f"Current best fitness: {best_fitness_ever:.6f}", flush=True)
        formatted = ", ".join(f"{x:.3f}" for x in best_solution_ever)
        print(f"Current best solution: [{formatted}]", flush=True)

    if pool is not None:
        pool.close()
        pool.join()

    return best_solution_ever, best_fitness_ever, fitness_tracker


# ----------------------------
# 5. Run GA, plot, save, final demo
# ----------------------------
if __name__ == "__main__":
    try:
        # Run the genetic algorithm
        t0 = time.perf_counter()
        best_solution, best_fitness, fitness_tracker = genetic_algorithm()
        elapsed = time.perf_counter() - t0

        print(f"Best Solution: {best_solution}")
        print(f"Best Fitness: {best_fitness:.6f}")
        print(f"Training time end (seconds): {elapsed:.2f}")

        # Plot fitness progression
        gens = np.arange(len(fitness_tracker))
        fig, ax = plt.subplots()
        ax.plot(gens, fitness_tracker, label="Best Fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Genetic Algorithm Progress")
        ax.legend()
        plt.show()  # Retain original plotting behavior

        # Save results to text file (as in original code)
        filename = "genetic_algorithm_results.txt"
        exists = os.path.exists(filename)
        with open(filename, "a") as f:
            if exists:
                f.write("\n\n\n")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Genetic Algorithm Results at {now}\n")
            f.write("Parameters:\n")
            f.write(f"  Population Size - {POPULATION_SIZE}\n")
            f.write(f"  Generations     - {MAX_GENERATIONS}\n")
            f.write(f"  Crossover Rate  - {CROSSOVER_RATE_BASE}\n")
            f.write(f"  Mutation Rate   - {MUTATION_RATE_BASE}\n")
            f.write(f"  Tournament K    - {TOURNAMENT_K}\n\n")
            f.write("Best Solution:\n")
            f.write(f"[{', '.join(map(str, best_solution.tolist()))}]\n\n")
            f.write("Best Fitness:\n")
            f.write(f"{best_fitness}\n")

        input("Press Enter to continue...")

        # Final demonstration run with visualization
        parser = argparse.ArgumentParser(description="Kessler Game Scenario Runner")
        parser.add_argument("--scenario", choices=scenarios.keys(),
                            default="random_repeatable",
                            help="Select a scenario")
        parser.add_argument("--game_type",
                            choices=["KesslerGame", "TrainerEnvironment"],
                            default="KesslerGame",
                            help="Visualization or fast sim")
        args = parser.parse_args()
        selected = scenarios[args.scenario]
        if args.game_type == "KesslerGame":
            game = KesslerGame(settings=game_settings)
        else:
            game = TrainerEnvironment(settings=game_settings)

        print(f"Running final scenario: {selected.name}")
        t1 = time.perf_counter()
        score, perf_data = game.run(best_solution,
                                    scenario=selected,
                                    controllers=[FuzzyController()])
        print(f"Final run completed in {time.perf_counter() - t1:.2f} seconds.")
    except Exception as e:
        # Catch any top level GA or demo error and log full traceback
        logger.error("Uncaught exception in main: %s", str(e))
        traceback.print_exc()
        raise
