import random
import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
from kesslergame import GraphicsType, KesslerGame, Scenario, TrainerEnvironment
import multiprocessing


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

# ----------------------------
# 3. GA Components
# ----------------------------

def create_random_individual(size):
    """
    Inputs:
        size - Integer - Number of genes in the individual
    Returns:
        A 1D NumPy array of numbers (0, 0.1, 0.2, …, 0.9) of length size.
    Purpose:
        Creates a new individual for the population with discrete gene values.
    """
    # Generate random integers in [0,9] and scale by 0.1
    return np.random.random(size)


def selection(population, fitnesses, K):
    """
    Inputs:
        population - List of 1D NumPy arrays (individuals)
        fitnesses - List of fitness values corresponding to each individual
        K - Integer - Number of individuals to sample for selection
    Returns:
        The best individual among K randomly selected individuals.
    Purpose:
        Selects a parent from the population using tournament selection.
    """
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
        parent1 - 1D NumPy array - First parent for crossover
        parent2 - 1D NumPy array - Second parent for crossover
        CROSSOVER_RATE - Float - Probability of crossover occurring
    Returns:
        child1, child2 - Two 1D NumPy arrays resulting from the crossover.
    Purpose:
        Mixes two parents to create two children by splicing at a random
        crossover point.
    """
    if random.random() < CROSSOVER_RATE:
        cp = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:cp], parent2[cp:]))
        child2 = np.concatenate((parent2[:cp], parent1[cp:]))
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    return child1, child2


def mutate(parent, MUTATION_RATE):
    """
    Inputs:
        parent - 1D NumPy array - Parent to mutate
        MUTATION_RATE - Float - Probability of mutating a chromosome (value)
    Returns:
        child - 1D NumPy array - Mutated child
    Purpose:
        Adds variation to a parent by potentially mutating each individual gene.
        The mutation now picks a value from {0, 0.1, ..., 0.9}.
    """
    child = parent.copy()  # make a copy of the parent's values
    if random.random() < MUTATION_RATE:
        for i in range(len(child)):
            if random.random() < MUTATION_RATE:
                child[i] = random.random()  # new gene is one of 0, 0.1, ..., 0.9
    return child

# ----------------------------
# 4. Main GA Loop
# ----------------------------

def genetic_algorithm(CHROMOSOME_SIZE=80,POPULATION_SIZE=2, MAX_GENERATIONS=200, mutation_rate=0.2, crossover_rate=0.7, K=10):
    """
    Inputs:
        POPULATION_SIZE - Integer: Number of individuals in the population
        MAX_GENERATIONS - Integer: Number of generations to run the GA for
        MUTATION_RATE   - Float: Probability of mutation per gene
        CROSSOVER_RATE  - Float: Probability of crossover occurring
        K - Integer: Tournament size for selection
    Returns:
        best_solution - 1D NumPy array: Best solution found by the GA
        best_fitness - Float: Fitness of the best solution
    Purpose:
        Optimizes the Membership Functions for the Kessler Game using a Genetic Algorithm.
    """

    
    # 1. Initialize population
    population = [create_random_individual(CHROMOSOME_SIZE) for _ in range(POPULATION_SIZE)]
    best_solution_ever = None
    best_fitness_ever = float('-inf')
    
    current_best_fit = None
    last_fitness = None
    fitness_age = 0
    
    for generation in range(MAX_GENERATIONS):
        # 2. Evaluate fitness for each individual

        gen_start_time = time.perf_counter()  # For performance tracking of each generation

        mutation_rate = MUTATION_RATE - 0.9*(MUTATION_RATE * (generation / MAX_GENERATIONS))  # Decay mutation rate over generations
        crossover_rate = CROSSOVER_RATE + (CROSSOVER_INCREASE - CROSSOVER_RATE) * (generation / MAX_GENERATIONS)  # Increase crossover rate over generations

        with multiprocessing.Pool(processes=4) as pool:
            fitnesses = pool.map(fitness_function, population)
            
        # Track the best in the current generation
        current_best_fit = max(fitnesses)
        current_best_ind = population[fitnesses.index(current_best_fit)]
        
        if current_best_fit > best_fitness_ever:
            best_fitness_ever = current_best_fit
            best_solution_ever = current_best_ind.copy()
        
        print(f"Generation {generation}, Best Fitness this generation{current_best_fit:.6f}, Best Fitness so far: {best_fitness_ever:.6f}")
        print(f"Current Best Individual: {best_solution_ever}")  # For debugging purposes
        # 3. Generate new population
        new_population = []

        new_population.append(current_best_ind.copy())  # Elitism: carry forward the best individual to the next generation

        if current_best_fit == last_fitness:
            fitness_age = fitness_age + 1
        else:
            fitness_age = 0
            
        if fitness_age >= 20:
            mutation_rate = MUTATION_RATE + (1 - MUTATION_RATE) * (fitness_age / 100)
            print(mutation_rate)

        if mutation_rate > 0.9:
            mutation_rate = 0.9

        while len(new_population) < POPULATION_SIZE:
            # 4. Selection (tournament selection)
            parent1 = selection(population, fitnesses, K)
            parent2 = selection(population, fitnesses, K)
            
            # 5. Crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            
            # 6. Mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.append(child1)
            new_population.append(child2)
        
        # In case we've exceeded the population size, trim the extra individuals
        population = new_population[:POPULATION_SIZE]

        last_fitness = current_best_fit  # Save the last best fitness for comparison in the next generation

        print(f"Generation {generation} completed in {time.perf_counter() - gen_start_time:.2f} seconds.")
    
    # Evaluate final population
    final_fitnesses = [fitness_function(ind) for ind in population]
    best_index = final_fitnesses.index(max(final_fitnesses))
    final_best_ind = population[best_index]
    
    # Return the overall best solution found
    if fitness_function(final_best_ind) > best_fitness_ever:
        best_solution_ever = final_best_ind.copy()
        best_fitness_ever = fitness_function(final_best_ind)
    
    return best_solution_ever, best_fitness_ever


















# ----------------------------
# 5. Fitness Function and Running the GA
# ----------------------------

def fitness_function(chromosome):

    """
    Inputs:
        chromosome - 1D NumPy array - Chromosome to evaluate
    Returns:
        fitness - Float - Fitness of the chromosome
    Purpose:
        Plays the Kessler Game with the given chromosome and returns the fitness
        based on game performance.
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
        help="The type of game to run. KesslerGame for visualization, TrainerEnvironment for fast, no-graphics simulation.",
    )

    args = parser.parse_args()

    selected_scenario: Scenario = scenarios[args.scenario]

    match args.game_type:
        case "KesslerGame":
            game = KesslerGame(settings=game_settings)
        case "TrainerEnvironment":
            game = TrainerEnvironment(settings=game_settings)

    fitness_sum = 0
    
    for _ in range(5):
        initial_time = time.perf_counter()
        score, perf_data = game.run(
            chromosome, scenario=selected_scenario, controllers=[FuzzyController()]
        )

        for team in score.teams:
            accuracy = team.accuracy
            deaths = team.deaths
            asteroids_hit = team.asteroids_hit


        fitness_sum += asteroids_hit - 10 * deaths**3

    return fitness_sum












if __name__ == '__main__':

    # Run the Genetic Algorithm to find the best solution



    purpose = "Changed to use any float instead of 0, 0.1, 0.2, ... 0.9"

    CHROMOSOME_SIZE = 68  # Number of genes in each individual
    POPULATION_SIZE = 30
    MAX_GENERATIONS = 30
    MUTATION_RATE   = 0.35
    MUTATION_DECAY = 0.90
    CROSSOVER_RATE  = 0.7
    CROSSOVER_INCREASE = 0.95 
    K = 4   


    start_time = time.perf_counter()
    best_solution, best_fitness = genetic_algorithm(CHROMOSOME_SIZE,POPULATION_SIZE, MAX_GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, K)

    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness:.6f}")

    print("Training time end (seconds): ", str(time.perf_counter() - start_time))



    filename = "genetic_algorithm_results.txt"
    file_exists = os.path.exists(filename)

    with open(filename, 'a') as file:
        # If file already exists, add a few blank lines first.
        if file_exists:
            file.write("\n\n\n")

        # Get current time and date
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write the header with time/date and the parameters/results
        file.write(f"Genetic Algorithm Results at {current_time}\n")
        file.write("Genetic Algorithm Parameters:\n")
        file.write(f"Population Size - {POPULATION_SIZE}\n")
        file.write(f"Generations - {MAX_GENERATIONS}\n")
        file.write(f"Crossover Rate - {CROSSOVER_RATE}\n")
        file.write(f"Mutation Rate - {MUTATION_RATE}\n")
        file.write(f"K - {K}\n")
        file.write("\n")
        file.write("Best Solution\n")
        file.write(f"[{', '.join(map(str, best_solution.tolist() if hasattr(best_solution, 'tolist') else best_solution))}]\n\n")
        file.write("Best Fitness\n")
        file.write(f"{best_fitness}\n")






    input("Press Enter to continue...")

    # ----------------------------
    # Run a final game with the best solution found
    # ----------------------------

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
        help="The type of game to run. KesslerGame for visualization, TrainerEnvironment for fast, no-graphics simulation.",
    )

    args = parser.parse_args()
    selected_scenario: Scenario = scenarios[args.scenario]

    match args.game_type:
        case "KesslerGame":
            game = KesslerGame(settings=game_settings)
        case "TrainerEnvironment":
            game = TrainerEnvironment(settings=game_settings)

    logger.info(f"Running scenario: {selected_scenario.name}")
    initial_time = time.perf_counter()
    score, perf_data = game.run(
        best_solution, scenario=selected_scenario, controllers=[FuzzyController()]
    )
