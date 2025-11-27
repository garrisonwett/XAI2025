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
INDIVIDUAL_WARN_SECONDS = 60.0
GAME_TIME_LIMIT = 60.0
EPSILON = 1e-6

# All Available Settings
game_settings = {
    "frequency": 30,
    "perf_tracker": True,
    "prints_on": True,
    "graphics_type": GraphicsType.Tkinter,
    "graphics_obj": None,
    "realtime_multiplier": 1,
    "time_limit": GAME_TIME_LIMIT,
    "random_ast_splits": False,
    "UI_settings": {
        "ships": True,
        "lives_remaining": True,
        "accuracy": True,
        "asteroids_hit": True,
        "shots_fired": True,
        "bullets_remaining": True,
        "controller_name": True,
    },
}

# ----------------------------
# 1. Hyperparameters
# ----------------------------
CHROMOSOME_SIZE     = 68
POPULATION_SIZE     = 50
MAX_GENERATIONS     = 3000
MUTATION_RATE_BASE  = 0.5
CROSSOVER_RATE_BASE = 0.8
CROSSOVER_INCREASE  = 0.9
TOURNAMENT_K        = 3
POOL_PROCESSES      = max(1, multiprocessing.cpu_count() - 4)
USE_MULTIPROCESSING = True
STOP_HOURS          = 0.3

# ----------------------------
# 2. Prepare simulators once
# ----------------------------
scenario_names = ["training2"]
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
    """
    vals = np.random.random(size).astype(float)
    vals = np.clip(vals, EPSILON, 1.0 - EPSILON)
    return vals


def fitness_function(chromosome):
    """
    Compute the fitness of one individual by simulating each scenario.

    Returns the sum over scenarios of:
        accuracy * asteroids_hit
        minus 30 * deaths^3
        minus team.mean_eval_time * 100
    """
    total = 0.0
    try:
        for name in scenario_names:
            score, _ = games[name].run(
                chromosome,
                scenario=scenarios[name],
                controllers=[FuzzyController()],
            )

            team = score.teams[0]
            msg = (
                f"Scenario: {name}, Asteroids Hit: {team.asteroids_hit}, "
                f"Deaths: {team.deaths}, Accuracy: {team.accuracy:.4f}, "
                f"Mean Eval Time: {team.mean_eval_time:.4f}s"
            )
            #logger.info(msg)

            total += (
                team.accuracy * team.asteroids_hit
                - 30 * (team.deaths**3)
                - team.mean_eval_time * 100
            )
    except Exception as e:
        logger.error("Exception inside fitness_function: %s", str(e))
        traceback.print_exc()
        raise
    return total


def guarded_fitness_evaluation(idx, chromosome):
    """
    Wrapper around fitness_function that:
      - Times the evaluation and warns if it is slow.
      - Catches and logs any exception with index and chromosome snippet.
    """
    start_time = time.perf_counter()

    try:
        fitness = fitness_function(chromosome)
    except Exception as e:
        logger.error(
            "Error during fitness evaluation for individual %d: %s",
            idx,
            str(e),
        )
        traceback.print_exc()
        chrom_preview = ", ".join(f"{g:.3f}" for g in chromosome[:10])
        logger.error(
            "Failed individual index %d, first 10 genes: [%s]",
            idx,
            chrom_preview,
        )
        raise

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Finished fitness eval for individual %d in %.3f seconds (fitness=%.6f)",
        idx,
        elapsed,
        fitness,
    )

    if elapsed > INDIVIDUAL_WARN_SECONDS:
        warn_msg = (
            f"Fitness eval for individual {idx} took {elapsed:.2f} seconds "
            f"(over {INDIVIDUAL_WARN_SECONDS} s). Likely culprit for perceived freeze."
        )
        logger.warning(warn_msg)

    return fitness


def tournament_selection(population, fitnesses, k):
    """
    Select one parent via tournament selection.
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
    With probability `rate`, perform one point crossover between two parents.
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
    Per gene mutation with probability `rate`, bounded to keep genes in (0, 1).
    """
    mask = np.random.rand(parent.size) < rate

    low  = np.maximum(-distance,      -parent)
    high = np.minimum( distance, 1.0 - parent)

    u = np.random.rand(parent.size)
    noise = low + u * (high - low)

    mutated = parent + noise
    mutated = np.clip(mutated, EPSILON, 1.0 - EPSILON)

    return np.where(mask, mutated, parent)


# ----------------------------
# 4. Main GA loop
# ----------------------------
def genetic_algorithm():
    """
    Runs the GA with dynamic rates, elitism, stagnation boost,
    optional parallel fitness evaluation, and a time based stop criterion.
    """
    population = [create_random_individual(CHROMOSOME_SIZE)
                  for _ in range(POPULATION_SIZE)]
    best_solution_ever = None
    best_fitness_ever = -np.inf
    last_fitness = None
    fitness_age = 0
    fitness_tracker = []

    ga_start_time = time.perf_counter()

    if USE_MULTIPROCESSING and POOL_PROCESSES > 1:
        logger.info(
            "Using multiprocessing with %d worker processes.",
            POOL_PROCESSES,
        )
        pool = multiprocessing.Pool(POOL_PROCESSES)
    else:
        logger.info("Running fitness evaluations in a single process.")
        pool = None

    for generation in range(MAX_GENERATIONS):
        elapsed_hours = (time.perf_counter() - ga_start_time) / 3600.0
        if elapsed_hours >= STOP_HOURS:
            logger.warning(
                "Reached GA time limit of %f hours at generation %d.",
                STOP_HOURS,
                generation,
            )
            break

        logger.info("Starting generation %d", generation)

        mutation_rate = MUTATION_RATE_BASE * (1 - generation / MAX_GENERATIONS)
        crossover_rate = (
            CROSSOVER_RATE_BASE
            + (CROSSOVER_INCREASE - CROSSOVER_RATE_BASE) * (generation / 300)
        )
        crossover_rate = min(crossover_rate, CROSSOVER_INCREASE)

        if fitness_age >= 20:
            boosted = MUTATION_RATE_BASE + (1 - MUTATION_RATE_BASE) * (fitness_age / 100)
            mutation_rate = min(boosted, 0.7)
            logger.info(
                "Boosted mutation rate to %.4f due to stagnation.",
                mutation_rate,
            )

        if pool is not None:
            fitnesses = pool.map(fitness_function, population)
        else:
            fitnesses = [
                guarded_fitness_evaluation(i, ind)
                for i, ind in enumerate(population)
            ]

        current_best_fit = max(fitnesses)
        current_best_ind = population[fitnesses.index(current_best_fit)]
        if current_best_fit > best_fitness_ever:
            best_fitness_ever = current_best_fit
            best_solution_ever = current_best_ind.copy()

        logger.info(
            "Generation %d complete. Best fitness this gen: %.6f, best ever: %.6f",
            generation,
            current_best_fit,
            best_fitness_ever,
        )

        fitness_tracker.append(best_fitness_ever)

        new_population = [current_best_ind.copy()]
        if current_best_fit == last_fitness:
            fitness_age += 1
        else:
            fitness_age = 0

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

        # The only per generation print:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_chrom = ", ".join(f"{x:.3f}" for x in current_best_ind)
        print(
            f"Gen {generation} | best_fitness={current_best_fit:.6f} | "
            f"chromosome=[{formatted_chrom}] | time={now_str}",
            flush=True,
        )

    if pool is not None:
        pool.close()
        pool.join()

    return best_solution_ever, best_fitness_ever, fitness_tracker


# ----------------------------
# 5. Run GA, plot, save, final demo
# ----------------------------
if __name__ == "__main__":
    try:
        t0 = time.perf_counter()
        best_solution, best_fitness, fitness_tracker = genetic_algorithm()
        elapsed = time.perf_counter() - t0

        print(f"Best Solution: {best_solution}")
        print(f"Best Fitness: {best_fitness:.6f}")
        print(f"Training time end (seconds): {elapsed:.2f}")

        gens = np.arange(len(fitness_tracker))
        fig, ax = plt.subplots()
        ax.plot(gens, fitness_tracker, label="Best Fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Genetic Algorithm Progress")
        ax.legend()
        plt.show()

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

        parser = argparse.ArgumentParser(description="Kessler Game Scenario Runner")
        parser.add_argument(
            "--scenario",
            choices=scenarios.keys(),
            default="training2",
            help="Select a scenario",
        )
        parser.add_argument(
            "--game_type",
            choices=["KesslerGame", "TrainerEnvironment"],
            default="KesslerGame",
            help="Visualization or fast sim",
        )
        args = parser.parse_args()
        selected = scenarios[args.scenario]
        if args.game_type == "KesslerGame":
            game = KesslerGame(settings=game_settings)
        else:
            game = TrainerEnvironment(settings=game_settings)

        print(f"Running final scenario: {selected.name}")
        t1 = time.perf_counter()
        score, perf_data = game.run(
            best_solution,
            scenario=selected,
            controllers=[FuzzyController()],
        )
        print(f"Final run completed in {time.perf_counter() - t1:.2f} seconds.")
    except Exception as e:
        logger.error("Uncaught exception in main: %s", str(e))
        traceback.print_exc()
        raise
