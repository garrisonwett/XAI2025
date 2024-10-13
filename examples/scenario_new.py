# -*- coding: utf-8 -*-

import time

from kesslergame import Scenario, KesslerGame, GraphicsType
from src.FuzzyController import FuzzyController


scenario_random_repeatable = Scenario(
    name="random_repeatable",
    num_asteroids=20,
    ship_states=[
        {"position": (400, 400), "angle": 90, "lives": 3, "team": 1}
    ],
    map_size=(1000, 800),
    time_limit=60,
    ammo_limit_multiplier=0.0,
    stop_if_no_ammo=False,
)

scenario_asteroid_wall = Scenario(
    name="asteroid_wall",
    asteroid_states=[
        {"position": (100, 200), "angle": 60.0, "speed": 40},
        {"position": (200, 200), "angle": 65.0, "speed": 40},
        {"position": (300, 200), "angle": 70.0, "speed": 40},
        {"position": (400, 200), "angle": 75.0, "speed": 40},
        {"position": (500, 200), "angle": 80.0, "speed": 40},
        {"position": (600, 200), "angle": 90.0, "speed": 40},
        {"position": (700, 200), "angle": 100.0, "speed": 40},
        {"position": (800, 200), "angle": 110.0, "speed": 40},
        {"position": (900, 200), "angle": 120.0, "speed": 40},
    ],
    ship_states=[{"position": (500, 600)}],
)

scenario_asteroid_wall_across = Scenario(
    name="asteroid_wall_across",
    asteroid_states=[
        {"position": (200, 200), "angle": 0.0, "speed": 180},
        {"position": (300, 200), "angle": 0.0, "speed": 180},
        {"position": (400, 200), "angle": 0.0, "speed": 180},
        {"position": (500, 200), "angle": 0.0, "speed": 180},
        {"position": (600, 200), "angle": 0.0, "speed": 180},
        {"position": (100, 200), "angle": 0.0, "speed": 180},
        {"position": (700, 200), "angle": 0.0, "speed": 180},
        {"position": (800, 200), "angle": 0.0, "speed": 180},
    ],
    ship_states=[{"position": (600, 700)}],
    ammo_limit_multiplier = (1/16)
)


# It fucking Sucks at this one lol
scenario_diversion = Scenario(
    name="diversion",
    asteroid_states=[
        {"position": (200, 200), "angle": 110.0, "speed": 40},
        {"position": (300, 200), "angle": 110.0, "speed": 40},
        {"position": (500, 100), "angle": 90.0, "speed": 40},
        {"position": (700, 200), "angle": 70.0, "speed": 40},
        {"position": (800, 200), "angle": 70.0, "speed": 40},
    ],
    ship_states=[{"position": (500, 500)}],
)

scenario_cross = Scenario(
    name="cross",
    asteroid_states=[
        {"position": (400, 700), "angle": 271.0, "speed": 40},
        {"position": (400, 100), "angle": 91.0, "speed": 40},
        {"position": (800, 400), "angle": 181.0, "speed": 40},
        {"position": (100, 400), "angle": 1.0, "speed": 40},
    ],
    ship_states=[{"position": (400, 400)}],

)

scenario_tracking_test = Scenario(
    name="tracking_test",
    asteroid_states=[
        {"position": (100, 200), "angle": 0.0, "speed": 100},
    ],
    ship_states=[{"position": (500, 600)}],
)

game_settings = {
    "perf_tracker": True,
    "graphics_mode": GraphicsType.Tkinter,
    "realtime_multiplier": 1,
}
game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

pre = time.perf_counter()
chrom = [2, 2, 2, 0, 1, 1, 0, 0, 1, 2, 2, 2, 0, 1, 1, 0, 0, 1]
chrom = [int(x) for x in chrom]
score, perf_data = game.run(
    scenario=scenario_random_repeatable,
    controllers=[FuzzyController()]
)

print("Scenario eval time: " + str(time.perf_counter() - pre))
print(score.stop_reason)
print("Asteroids hit: " + str([team.asteroids_hit for team in score.teams]))
print("Deaths: " + str([team.deaths for team in score.teams]))
print("Accuracy: " + str([team.accuracy for team in score.teams]))
print("Mean eval time: " + str([team.mean_eval_time for team in score.teams]))
