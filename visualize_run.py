import argparse
import os
import sys
import numpy as np
import time
import pandas as pd  # Added for data analysis

from kesslergame import GraphicsType, KesslerGame, Scenario, TrainerEnvironment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# these imports are specific to your project structure
from redone_controller import FuzzyController
from scenarios import scenarios, random_repeatable_frozen
from utils import LoggerUtility, LoggingLevel
from algorithms import load_chromosome

# Set up the logger
logger = LoggerUtility(LoggingLevel.DEBUG).get_logger()

# Game settings
game_settings = {
    "frequency": 30,
    "perf_tracker": True,
    "prints_on": True,
    "graphics_type": GraphicsType.Tkinter,
    "graphics_obj": None,
    "realtime_multiplier": 1,
    "time_limit": float("inf"),
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kessler Game Scenario Runner")

    parser.add_argument(
        "--scenario",
        choices=scenarios.keys(),
        type=str,
        default="training2",
        help="Select a scenario by name: " + ", ".join(scenarios.keys()),
    )

    parser.add_argument(
        "--game_type",
        choices=["KesslerGame", "TrainerEnvironment"],
        type=str,
        default="KesslerGame",
        help="KesslerGame for visualization, or TrainerEnvironment for no-graphics simulation.",
    )

    parser.add_argument(
        "--chromosome_file",
        type=str,
        default="best_kessler_fuzzy.pkl",
        help="Path to the saved GA chromosome pickle.",
    )

    args = parser.parse_args()

    # Load scenario
    selected_scenario: Scenario = scenarios[args.scenario]

    # Load chromosome
    if not os.path.isfile(args.chromosome_file):
        raise FileNotFoundError(f"Could not find file: {args.chromosome_file}")

    chromosome = load_chromosome(args.chromosome_file)
    print(f"Loaded chromosome from: {args.chromosome_file}")

    # Select game mode
    match args.game_type:
        case "KesslerGame":
            game = KesslerGame(settings=game_settings)
        case "TrainerEnvironment":
            game = TrainerEnvironment(settings=game_settings)

    logger.info(f"Running scenario: {selected_scenario.name}")
    initial_time = time.perf_counter()

    # Deterministic asteroid layout for consistent playback
    # gen = np.random.randint(0, 1000000)
    # selected_scenario = random_repeatable_frozen(gen)

    # Run game with the controller using the loaded chromosome
    score, perf_data = game.run(
        scenario=selected_scenario,
        controllers=[FuzzyController(chromosome)]
    )

    print("Total scenario eval time:", time.perf_counter() - initial_time)
    print("Stop reason:", score.stop_reason)
    print("Asteroids hit:", [team.asteroids_hit for team in score.teams])
    print("Deaths:", [team.deaths for team in score.teams])
    print("Accuracy:", [team.accuracy for team in score.teams])
    print("Mean eval time:", [team.mean_eval_time for team in score.teams])

    # --- PERFORMANCE DATA PARSING ---
    if perf_data:
        print("\n--- Performance Data Analysis ---")
        
        # 1. Convert to DataFrame
        df = pd.DataFrame(perf_data)
        df['frame'] = df.index

        # 2. Extract Controller Times
        # Safely handle cases where controller_times might be empty or missing
        if 'controller_times' in df.columns and len(df) > 0:
            controller_data = pd.DataFrame(df['controller_times'].tolist())
            # Rename columns to Ship 1, Ship 2, etc.
            controller_data.columns = [f'Ship {i+1}' for i in range(controller_data.shape[1])]
            controller_data['frame'] = df.index
            
            print("\nController Execution Times (ms):")
            # Multiply by 1000 for readable ms values
            print((controller_data.drop('frame', axis=1) * 1000).head())
            print(f"Avg Controller Time: {(df['total_controller_time'].mean() * 1000):.3f} ms")

        print("\nFrame Performance Breakdown (Head):")
        print(df[['frame', 'physics_update', 'graphics_draw', 'total_frame_time']].head())