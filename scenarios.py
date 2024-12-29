from typing import Dict

from kesslergame.scenario import Scenario

scenarios: Dict[str, Scenario] = {
    "random_repeatable": Scenario(
        name="random_repeatable",
        num_asteroids=20,
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
        ship_states=[{"position": (500, 700), "angle": 270, "lives": 5, "team": 1}],
        asteroid_states=[{"position": (200, 200), "angle": 0.0, "speed": 150, "size": 4}],
        map_size=(1200, 900),
        time_limit=90,
        ammo_limit_multiplier=0.0,
        stop_if_no_ammo=False,
    ),
    # Add more scenarios here
}