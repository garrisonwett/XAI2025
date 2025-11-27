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


    chromosome = [0.10469125, 0.04297936, 0.90682691, 0.8567271, 0.96931874, 0.19946062,
        0.99222546, 0.36787889, 0.37465216, 0.57602967, 0.9741035,  0.45342712,
        0.96625418, 0.1381653,  0.86990328, 0.30157597, 0.22415513, 0.03560846,
        0.30005406, 0.51914667, 0.33582922, 0.61661478, 0.68968709, 0.02978293,
        0.11414158, 0.97052574, 0.11944187, 0.78552032, 0.61180064, 0.15267815,
        0.75501595, 0.64288643, 0.51537604, 0.22022223, 0.12289156, 0.56199781,
        0.60240319, 0.36187603, 0.70678016, 0.20017822, 0.62290751, 0.45291756,
        0.52157168, 0.81336963, 0.47832499, 0.25582713, 0.36338551, 0.29009963,
        0.19297389, 0.94645667, 0.19185161, 0.67609084, 0.51499505, 0.3095522,
        0.57000158, 0.74548439, 0.95348415, 0.99243167, 0.69706017, 0.8484516,
        0.99103867, 0.96123234, 0.86984809, 0.64808913, 0.60069154, 0.91684167,
        0.78842997, 0.69871571]
    
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
