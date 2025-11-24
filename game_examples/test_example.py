import argparse
import os
import sys
import numpy as np
import time

from kesslergame import GraphicsType, KesslerGame, Scenario, TrainerEnvironment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#from TeamTempNameSubmission.fuzzy_controller import FuzzyController
from redone_controller import FuzzyController
from scenarios import scenarios, random_repeatable_frozen
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
    initial_time = time.perf_counter()


    chromosome = [0.700, 0.701, 0.700, 0.700, -0.000, 0.600, 0.100, 0.500, 0.400, 0.100, 0.700, 0.800, 0.500, 0.600, 0.900, 0.000, 0.800, 0.400, 0.600, 0.800, 0.200, 0.200, 0.000, 0.300, 0.800, 0.000, 0.700, 0.400, 0.500, 0.100, 0.300, 0.200, 0.600, 0.100, 0.100, 0.800, 0.800, 0.900, 0.200, 0.300, 0.200, 0.100, 0.500, 0.699, 0.900, 0.100, 0.100, 0.200, 0.300, 0.800, 0.400, 0.700, 0.400, 0.400, 0.400, 0.000, 0.200, 0.700, 0.800, -0.000, 0.900, 0.100, 0.600, 0.100, 0.400, 0.600, 0.800, 0.800, 0.700, 0.500, -0.000, 0.700, 0.600, 0.200, 0.500, 0.800, 0.500, 0.700, 0.500, 0.500, 0.701, 0.400, 0.100, 0.000, 0.900, 0.800, 0.700, 0.500, 0.800, 0.000, 0.600, 0.400, -0.000, 0.000, 0.500, 0.000, 0.900, 0.801, 0.100, 0.400, 0.300, 0.000, 0.400, 0.700, 0.000, 0.500, 0.800, 0.700, 0.800, 0.400, 0.300, -0.000, 0.300, 0.600, 0.000, 0.300, 0.600, 0.600, 0.200, 0.700, 0.500, 0.300, 0.900, 0.500, 0.000, 0.300, 0.600, 0.500, 0.400, 0.200, 0.100, 0.400, 0.000, 0.500, 0.300, 0.000, 0.001, 0.400, 0.400, 0.800, 0.200, 0.900, 0.200, 0.900, 0.700, 0.000, 0.600, 0.300, 0.500, 0.300, 0.500, 0.100, 0.900, 0.300, 0.200, 0.900, 0.000, 0.900, 0.800, 0.700, 0.300, 0.600, 0.000, 0.600, 0.600, 0.800, 0.400, 0.300, 0.900, 0.300, -0.000, 0.700, 0.600, 0.700, 0.500, 0.800, 0.500]  # No chromosome needed for standard controllers

    gen = np.random.randint(0, 1000000)
    selected_scenario = random_repeatable_frozen(gen)
    
    score, perf_data = game.run(
        chromosome, scenario=selected_scenario, controllers=[FuzzyController()]
    )

    print("Total scenario eval time: ", str(time.perf_counter() - initial_time))
    print("Stop reason: ", score.stop_reason)
    print("Asteroids hit: ", str([team.asteroids_hit for team in score.teams]))
    print("Deaths: ", str([team.deaths for team in score.teams]))
    print("Accuracy: ", str([team.accuracy for team in score.teams]))
    print("Mean eval time: ", str([team.mean_eval_time for team in score.teams]))
