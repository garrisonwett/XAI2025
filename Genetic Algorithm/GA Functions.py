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
    An individual is a list [a, b, c, d], representing a cubic polynomial.
    We initialize each coefficient randomly in [COEFF_MIN, COEFF_MAX].
    """
    return_array = []
    for i in range(2):
        return_array = [*return_array, *random_ordered_sequence(3)]
    return return_array

def random_ordered_sequence(L):
    """Return an array of L floats in [0,1] sorted in ascending order."""
    sequence = [random.random() for _ in range(L)]
    return sorted(sequence)


def selection(population, fitnesses,K):
    """
    Select one individual from the population using tournament selection 
    (or you could implement roulette wheel, rank selection, etc.).
    Here, we pick K random individuals and choose the one with best fitness.
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
    Single-point crossover on the coefficients [a,b,c,d].
    If CROSSOVER_RATE is met, we swap coefficients at a random point.
    Otherwise, children are clones of the parents.
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
    For each coefficient, apply a random change with probability MUTATION_RATE.
    """
    if random.random() < MUTATION_RATE:
        # Add a small random value (could also do random reset or other variants)
        # Here we do a slight perturbation, say +/- up to 10% of current value
        child = random_ordered_sequence(len(parent))
    else:
        child = parent[:]
    return child

# ----------------------------
# 4. Main GA Loop
# ----------------------------

def genetic_algorithm(POPULATION_SIZE = 15,MAX_GENERATIONS = 20, MUTATION_RATE   = 0.1,CROSSOVER_RATE  = 0.9, K=10):
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
    fitness_sum = 0
    print("\n")
    print("New Fitness")
    for _ in range(7):
        print("New Game Run")
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