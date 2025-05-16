from typing import Dict
import numpy as np
from kesslergame.scenario import Scenario

scenarios: Dict[str, Scenario] = {
    "random_repeatable": Scenario(
        name="random_repeatable",
        num_asteroids=7,
        asteroid_states=None,
        ship_states=[{"position": (400, 400), "angle": 90, "lives": 3, "team": 1}],
        map_size=(1000, 800),
        time_limit=60,
        ammo_limit_multiplier=0.0,
        stop_if_no_ammo=False,
    ),
    "battle_arena": Scenario(
        name="battle_arena",
        num_asteroids=15,
        ship_states=[{"position": (200, 200), "angle": 45, "lives": 5, "team": 2}],
        map_size=(1200, 900),
        time_limit=90,
        ammo_limit_multiplier=0.5,
        stop_if_no_ammo=True,
    ),
    "one_asteroid": Scenario(
        name="one_asteroid",
        ship_states=[{"position": (250, 200), "angle": 0, "lives": 5, "team": 1,}],
        asteroid_states=[
            {"position": (200, 200), "angle": 0.0, "speed": 0.001, "size": 1}
        ],
        map_size=(1200, 900),
        time_limit=90,
        ammo_limit_multiplier=0.0,
        stop_if_no_ammo=False,
    ),
    "collision_test": Scenario(
        name="collision_test",
        ship_states=[{"position": (900, 225), "angle": 270, "lives": 5, "team": 1}],
        asteroid_states=[
            {"position": (200, 200), "angle": 0.0, "speed": 150, "size": 4}
        ],
        map_size=(1200, 900),
        time_limit=90,
        ammo_limit_multiplier=0.0,
        stop_if_no_ammo=False,
    ),
    "avoid": Scenario(
        name="avoid",
        ship_states=[{"position": (600, 200), "angle": 0, "lives": 5, "team": 1,}],
        asteroid_states=[
            {"position": (400, 200), "angle": 0.0, "speed": 150, "size": 3},
            {"position": (700, 200), "angle": 0.0, "speed": 150, "size": 3}
        ],
        map_size=(1200, 900),
        time_limit=90,
        ammo_limit_multiplier=0.0,
        stop_if_no_ammo=False,
    ),
    "aim_trainer": Scenario(
        name="one_asteroid",
        ship_states=[{"position": (600, 500), "angle": 359, "lives": 5, "team": 1,}],
        asteroid_states=[
            {"position": (200, 200), "angle": 0, "speed": 150, "size": 4},
            {"position": (500, 200), "angle": 0, "speed": 150, "size": 4},
            {"position": (800, 200), "angle": 0, "speed": 150, "size": 4}
        ],
        map_size=(1200, 900),
        time_limit=90,
        ammo_limit_multiplier=0.0,
        stop_if_no_ammo=False,
    ),
    "crush": Scenario(
    name="crush",
    asteroid_states=[
        {"position": (200, 200), "angle": 50.0, "speed": 40, "size": 4},
        {"position": (300, 200), "angle": 60.0, "speed": 40, "size": 4},
        {"position": (400, 200), "angle": 70.0, "speed": 40, "size": 4},
        {"position": (500, 200), "angle": 90.0, "speed": 40, "size": 4},
        
    ],
    ship_states=[{"position": (500, 500)}],
    ),

    "training1": Scenario(
    name="training1",
    ship_states=[{"position": (400, 400), "angle": 90, "lives": 5, "team": 1}],
    asteroid_states=[
        {"position": (100, 100), "angle": 10, "speed": 100, "size": 4},
        {"position": (200, 100), "angle": 20, "speed": 100, "size": 4},
        {"position": (300, 100), "angle": 30, "speed": 100, "size": 4},
        {"position": (400, 100), "angle": 40, "speed": 100, "size": 4},
        {"position": (500, 100), "angle": 50, "speed": 100, "size": 4},
        {"position": (600, 100), "angle": 60, "speed": 100, "size": 4},
        {"position": (700, 100), "angle": 70, "speed": 100, "size": 4},
        {"position": (800, 100), "angle": 80, "speed": 100, "size": 4},
        {"position": (100, 700), "angle": 90, "speed": 100, "size": 4},
        {"position": (200, 700), "angle": 100, "speed": 100, "size": 4},
        {"position": (300, 700), "angle": 110, "speed": 100, "size": 4},
        {"position": (400, 700), "angle": 120, "speed": 100, "size": 4},
        {"position": (500, 700), "angle": 130, "speed": 100, "size": 4},
        {"position": (600, 700), "angle": 140, "speed": 100, "size": 4},
        {"position": (700, 700), "angle": 150, "speed": 100, "size": 4},
        {"position": (800, 700), "angle": 160, "speed": 100, "size": 4},

    ],
    ),

        "training2": Scenario(
    name="training2",
    ship_states=[{"position": (400, 400), "angle": 90, "lives": 5, "team": 1}],
    asteroid_states=[
        {"position": (100, 100), "angle": 0, "speed": 100, "size": 4},
        {"position": (100, 200), "angle": 0, "speed": 100, "size": 4},
        {"position": (100, 300), "angle": 0, "speed": 100, "size": 4},
        {"position": (100, 400), "angle": 0, "speed": 100, "size": 4},
        {"position": (100, 500), "angle": 0, "speed": 100, "size": 4},
        {"position": (100, 600), "angle": 0, "speed": 100, "size": 4},
        {"position": (100, 700), "angle": 0, "speed": 100, "size": 4},

    ],
    ),

            "training3": Scenario(
    name="training3",
    ship_states=[{"position": (400, 400), "angle": 90, "lives": 5, "team": 1}],
    asteroid_states=[
        {"position": (100, 100), "angle": 45, "speed": 100, "size": 4},
        {"position": (700, 700), "angle": 225, "speed": 100, "size": 4},
        {"position": (100, 700), "angle": 315, "speed": 100, "size": 4},
        {"position": (700, 100), "angle": 135, "speed": 100, "size": 4},
    ],
    )

    # Add more scenarios here
}

