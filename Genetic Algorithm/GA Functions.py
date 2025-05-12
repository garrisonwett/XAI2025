import random
import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
from kesslergame import GraphicsType, KesslerGame, Scenario, TrainerEnvironment
import multiprocessing
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fuzzy_controller import FuzzyController
from scenarios import scenarios
from utils import LoggerUtility, LoggingLevel

# Set up the logger
logger = LoggerUtility(LoggingLevel.DEBUG).get_logger()

# All Available Settings
game_settings = {
    "frequency": 30,  # Dictates both frequency and time_step settings (float)
    "perf_tracker": True,  # (bool)
    "prints_on": True,  # (bool)
    "graphics_type": GraphicsType.Tkinter,  # (GraphicsType)
    "graphics_obj": None,  # (Optional[KesslerGraphics])
    "realtime_multiplier": 1,  # (float)
    "time_limit": float("inf"),  # (float)
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
CHROMOSOME_SIZE     = 68      # Number of genes per individual
POPULATION_SIZE     = 20      # How many individuals in each generation
MAX_GENERATIONS     = 100      # Maximum number of GA iterations
MUTATION_RATE_BASE  = 0.3     # Starting mutation probability per gene
CROSSOVER_RATE_BASE = 0.4     # Starting crossover probability per mating
CROSSOVER_INCREASE  = 0.9     # Final (max) crossover probability
TOURNAMENT_K        = 3       # Tournament size for parent selection
POOL_PROCESSES      = 8       # Number of worker processes for fitness eval

# ----------------------------
# 2. Prepare simulators once
# ----------------------------
scenario_names = ["training1", "training2", "training3"]
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
    Create a random chromosome of given length with discrete gene values.
    
    Each gene is drawn uniformly from {0.0, 0.1, …, 0.9}. We switched
    from continuous np.random.random to np.random.randint for speed
    and exact discreteness.
    """
    return (np.random.randint(0, 10, size) / 10.0).astype(float)


def fitness_function(chromosome):
    """
    Compute the fitness of one individual by simulating each scenario.

    Uses pre-instantiated TrainerEnvironment instances to avoid
    rebuilding parsers or environments on every call, which greatly
    reduced per-evaluation overhead.
    
    Returns the sum over scenarios of (asteroids_hit − 10·deaths³).
    """
    total = 0.0
    for name in scenario_names:
        score, _ = games[name].run(
            chromosome,
            scenario=scenarios[name],
            controllers=[FuzzyController()]
        )
        team = score.teams[0]
        total += team.asteroids_hit - 10 * (team.deaths**2)
    return total


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
    
    - A random cut point is chosen (excluding ends), and genes are
      spliced to create two children.
    - Otherwise, children are exact copies of their parents.
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
    shifts the gene by at most `distance` but always stays within [0,1].
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

    # 4) apply noise only where mask is True
    mutated = parent + noise
    return np.where(mask, mutated, parent)

# ----------------------------
# 4. Main GA loop
# ----------------------------
def genetic_algorithm():
    """
    Runs the GA with:
      - dynamic mutation/crossover rates (linear decay/increase)
      - elitism (carry forward the best individual)
      - stagnation-based mutation boost (if no improvement for 20 gens)
      - parallel fitness evaluation via a persistent multiprocessing.Pool
    
    Returns:
        best_solution_ever: numpy array of the top chromosome found
        best_fitness_ever:  fitness score of that chromosome
        fitness_tracker:   list of best_fitness_ever per generation
    """
    # 1) Initialize population
    population = [create_random_individual(CHROMOSOME_SIZE)
                  for _ in range(POPULATION_SIZE)]
    best_solution_ever = None
    best_fitness_ever = -np.inf
    last_fitness = None
    fitness_age = 0
    fitness_tracker = []

    # 2) Keep pool alive throughout all generations
    pool = multiprocessing.Pool(POOL_PROCESSES)
    for generation in range(MAX_GENERATIONS):
        gen_start = time.perf_counter()

        # 3) Update dynamic rates
        #    - mutation decays linearly from base → 0
        #    - crossover ramps linearly from base → CROSSOVER_INCREASE
        mutation_rate = MUTATION_RATE_BASE * (1 - generation / MAX_GENERATIONS)
        crossover_rate = CROSSOVER_RATE_BASE + \
                         (CROSSOVER_INCREASE - CROSSOVER_RATE_BASE) * \
                         (generation / MAX_GENERATIONS)

        # 4) Evaluate fitnesses in parallel
        fitnesses = pool.map(fitness_function, population)

        # 5) Track best solution
        current_best_fit = max(fitnesses)
        current_best_ind = population[fitnesses.index(current_best_fit)]
        if current_best_fit > best_fitness_ever:
            best_fitness_ever = current_best_fit
            best_solution_ever = current_best_ind.copy()

        # 6) Original logging for transparency
        print(f"Generation {generation}, "
              f"Best Fitness this generation {current_best_fit:.6f}, "
              f"Best Fitness so far: {best_fitness_ever:.6f}")
        print(f"Current Best Individual: {best_solution_ever}")

        fitness_tracker.append(best_fitness_ever)

        # 7) Elitism & stagnation boost
        new_population = [current_best_ind.copy()]
        if current_best_fit == last_fitness:
            fitness_age += 1
        else:
            fitness_age = 0

        if fitness_age >= 20:
            # Increase mutation to escape platesaus
            boosted = MUTATION_RATE_BASE + (1 - MUTATION_RATE_BASE) * (fitness_age / 100)
            mutation_rate = min(boosted, 0.7)
            print(f"Boosted mutation rate to {mutation_rate:.4f}")

        # 8) Create the rest of the new population

        distance = (1-generation)/MAX_GENERATIONS

        while len(new_population) < POPULATION_SIZE:
            p1 = tournament_selection(population, fitnesses, TOURNAMENT_K)
            p2 = tournament_selection(population, fitnesses, TOURNAMENT_K)
            c1, c2 = crossover(p1, p2, crossover_rate)
            new_population.append(mutate(c1, mutation_rate, distance))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(c2, mutation_rate, distance))

        population = new_population[:POPULATION_SIZE]
        last_fitness = current_best_fit

        print(f"Generation {generation} completed in "
              f"{time.perf_counter() - gen_start:.2f} seconds.\n")

    pool.close()
    pool.join()
    return best_solution_ever, best_fitness_ever, fitness_tracker

# ----------------------------
# 5. Run GA, plot, save, final demo
# ----------------------------
if __name__ == "__main__":
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
    ax.plot(gens, fitness_tracker, label='Best Fitness')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Genetic Algorithm Progress')
    ax.legend()
    plt.show()  # Retain original plotting behavior

    # Save results to text file (as in original code)
    filename = "genetic_algorithm_results.txt"
    exists = os.path.exists(filename)
    with open(filename, 'a') as f:
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