import random
# ----------------------------
# 1. Problem Definition
# ----------------------------
import argparse
import os
import sys
import time

from kesslergame import GraphicsType, KesslerGame, Scenario, TrainerEnvironment

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
# 2. GA Parameters
# ----------------------------

POPULATION_SIZE = 1
MAX_GENERATIONS = 20
MUTATION_RATE   = 0.1
CROSSOVER_RATE  = 0.9

# ----------------------------
# 3. GA Components
# ----------------------------

def create_random_individual():
    """
    Inputs:
    Something to tell it how many genes to make, how many sets, and what those sets should be

    Returns:
    A random array of floats in [0,1] of length L OR an List of Lists with variables [0,1]

    Purpose:
    Creates a new individial for the population in the GA
    """
    return_array = []
    for i in range(2):
        return_array = [*return_array, *random_ordered_sequence(3)]
    return return_array

def random_ordered_sequence(L):
    """Return an array of L floats in [0,1] sorted in ascending order.
    
    This probably doesnt need to be its own function, unless we want to use it for mutation, TBD
    
    Inputs:
    L - Integer - Length of the sequence to be created
    
    Returns - A list of floats in [0,1] sorted in ascending order
    
    Purpose - Creates a random sequence of floats in [0,1] sorted in ascending order. 
              This is for Membership Functions in the Fuzzy Inference Systems
    """
    
    
    sequence = [random.random() for _ in range(L)]
    return sorted(sequence)


def selection(population, fitnesses,K):
    """
    Inputs:
    population - List of individuals in the population
    fitnesses - List of fitnesses for each individual in the population
    K - Integer - Number of individuals to select 

    Returns:
    The best found individual in this group of K individuals

    Purpose:
    Selects the best individual from a random group of K individuals in the population. This is used to select parents for crossover.

    """
    K = 10
    best_individual = None
    best_fitness = float('-inf')
    
    for _ in range(K):
        idx = random.randint(0, len(population) - 1)
        if fitnesses[idx] > best_fitness:
            best_fitness = fitnesses[idx]
            best_individual = population[idx]
    return best_individual

def crossover(parent1, parent2, CROSSOVER_RATE):
    """
    Inputs:
    parent1 - List of floats - First parent for crossover
    parent2 - List of floats - Second parent for crossover
    CROSSOVER_RATE - Float - Probability of crossover occurring

    Returns:
    child1 - List of floats - First child from crossover
    child2 - List of floats - Second child from crossover

    Purpose:
    Mixes two parents to create two children. This is used to create new individuals in the population.

    TODO: Needs to be only doing crossovers where it is acceptable to do so - i.e. only where a set of MFs start/stop not in the middle of them.


    """
    if random.random() < CROSSOVER_RATE:
        # Valid crossover points: after 1st, 2nd, or 3rd coefficient
        cp = random.randint(1, len(parent1) - 1)
        child1 = parent1[:cp] + parent2[cp:]
        child2 = parent2[:cp] + parent1[cp:]
    else:
        child1, child2 = parent1[:], parent2[:]
    return child1, child2

def mutate(parent, MUTATION_RATE):
    """
    Inputs:
    parent - List of floats - Parent to mutate
    MUTATION_RATE - Float - Probability of mutation occurring

    Returns:
    child - List of floats - Mutated child

    Purpose:
    Adds variation to a parent to create a child. This is used to create new individuals in the population.
    
    """
    if random.random() < MUTATION_RATE:
        child = create_random_individual()

    else:
        child = parent[:]
    return child

# ----------------------------
# 4. Main GA Loop
# ----------------------------

def genetic_algorithm(POPULATION_SIZE = 15, MAX_GENERATIONS = 20, MUTATION_RATE   = 0.1, CROSSOVER_RATE  = 0.9, K=10):
    """
    Inputs:
    POPULATION_SIZE - Integer - Number of individuals in the population
    MAX_GENERATIONS - Integer - Number of generations to run the GA for
    MUTATION_RATE   - Float - Probability of mutation occurring
    CROSSOVER_RATE  - Float - Probability of crossover occurring
    K - Integer - Number of individuals to select


    Returns:
    best_solution - List of floats - Best solution found by the GA
    best_fitness - Float - Fitness of the best solution found by the GA

    Purpose:
    Optimizes the Membership Functions of the Fuzzy Inference System for the Kessler Game using a Genetic Algorithm.
    
    """
    
    
    
    
    
    # 1. Initialize population
    population = [create_random_individual() for _ in range(POPULATION_SIZE)]
    # Track best solution
    best_solution_ever = None
    best_fitness_ever = float('-inf')
    
    for generation in range(MAX_GENERATIONS):
        # 2. Evaluate fitness
        fitnesses = [fitness_function(ind) for ind in population]
        
        # Check if we found a better solution
        current_best_fit = max(fitnesses)
        current_best_ind = population[fitnesses.index(current_best_fit)]
        
        if current_best_fit > best_fitness_ever:
            best_fitness_ever = current_best_fit
            best_solution_ever = current_best_ind[:]
        
        # Print progress
        if generation % 1 == 0:
            print(f"Generation {generation}, Best Fitness: {best_fitness_ever:.6f}")

        # 3. Generate new population
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            # 4. Selection
            parent1 = selection(population, fitnesses, K)
            parent2 = selection(population, fitnesses, K)
            
            # 5. Crossover
            child1, child2 = crossover(parent1, parent2, CROSSOVER_RATE)
            
            # 6. Mutation
            child1 = mutate(child1, MUTATION_RATE)
            child2 = mutate(child2, MUTATION_RATE)
            
            new_population.append(child1)
            new_population.append(child2)

        # Replace old population
        population = new_population
    
    # Evaluate final population
    final_fitnesses = [fitness_function(ind) for ind in population]
    best_index = final_fitnesses.index(max(final_fitnesses))
    final_best_ind = population[best_index]
    
    # Possibly return whichever was truly best across all generations
    if fitness_function(final_best_ind) > best_fitness_ever:
        best_solution_ever = final_best_ind
        best_fitness_ever = fitness_function(final_best_ind)
    
    return best_solution_ever, best_fitness_ever

# ----------------------------
# 5. Running the GA
# ----------------------------



def fitness_function(chromosome):
    """
    Inputs:
    chromosome - List of floats - Chromosome to evaluate

    Returns:
    fitness - Float - Fitness of the chromosome


    Purpose:
    Plays the game with the given chromosome and returns the fitness of the chromosome based on the performance in the game.
    
    """
    parser = argparse.ArgumentParser(description="Kessler Game Scenario Runner")

    parser.add_argument(
        "--scenario",
        choices=scenarios.keys(),
        type=str,
        default="random_repeatable",
        help="Select a scenario by name: " + ", ".join(scenarios.keys()),
    )

    parser.add_argument(
        "--game_type",
        choices=["KesslerGame", "TrainerEnvironment"],
        type=str,
        default="TrainerEnvironment",
        help="The type of game to run. KesslerGame for visualization, TrainerEnvironment for max-speed, no-graphics simulation.",
    )

    args = parser.parse_args()

    selected_scenario: Scenario = scenarios[args.scenario]

    match args.game_type:
        case "KesslerGame":
            game = KesslerGame(settings=game_settings)
        case "TrainerEnvironment":
            game = TrainerEnvironment(settings=game_settings)

    #logger.info(f"Running scenario: {selected_scenario.name}")

    fitness_sum = 0
    print("\n")
    print("New Fitness")
    for _ in range(7):
        print("New Game Run")
        

        initial_time = time.perf_counter()
        score, perf_data = game.run(
            chromosome, scenario=selected_scenario, controllers=[FuzzyController()]
        )

        for team in score.teams:
            accuracy = team.accuracy
            deaths = team.deaths
            asteroids_hit = team.asteroids_hit
        fitness_sum = fitness_sum + (asteroids_hit * accuracy - 3 * deaths**5)

    return fitness_sum
    

best_solution, best_fitness = genetic_algorithm()
print(f"Best Solution: {best_solution}, Best Fitness: {best_fitness:.6f}")

input("Press Enter to continue...")

parser = argparse.ArgumentParser(description="Kessler Game Scenario Runner")

parser.add_argument(
    "--scenario",
    choices=scenarios.keys(),
    type=str,
    default="random_repeatable",
    help="Select a scenario by name: " + ", ".join(scenarios.keys()),
)

parser.add_argument(
    "--game_type",
    choices=["KesslerGame", "TrainerEnvironment"],
    type=str,
    default="KesslerGame",
    help="The type of game to run. KesslerGame for visualization, TrainerEnvironment for max-speed, no-graphics simulation.",
)

args = parser.parse_args()

selected_scenario: Scenario = scenarios[args.scenario]

match args.game_type:
    case "KesslerGame":
        game = KesslerGame(settings=game_settings)
    case "TrainerEnvironment":
        game = TrainerEnvironment(settings=game_settings)

logger.info(f"Running scenario: {selected_scenario.name}")
chromosome = None
initial_time = time.perf_counter()
score, perf_data = game.run(
    best_solution, scenario=selected_scenario, controllers=[FuzzyController()]
)




















